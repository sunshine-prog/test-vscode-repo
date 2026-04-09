from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import PchipInterpolator

from spray_defect.config import ensure_dir, load_yaml, save_csv, save_json
from spray_defect.trainer import train_and_evaluate
from spray_defect.trainer_v2 import train_and_evaluate_v2


AUGMENT_VARIANTS: dict[str, dict[str, Any]] = {
    "none": {
        "label": "No Augmentation",
        "label_zh": "无增强",
        "mode": "none",
        "methods": None,
    },
    "spatial": {
        "label": "Spatial",
        "label_zh": "空间增强",
        "mode": "spatial",
        "methods": ["spatial"],
    },
    "frequency": {
        "label": "Frequency",
        "label_zh": "频率增强",
        "mode": "frequency",
        "methods": ["frequency"],
    },
    "spatial_frequency": {
        "label": "Spatial + Frequency",
        "label_zh": "空间+频率增强",
        "mode": "spatial",
        "methods": ["spatial", "frequency"],
    },
}

AUGMENT_COLORS = {
    "none": "#596275",
    "spatial": "#6D597A",
    "frequency": "#5E8C61",
    "spatial_frequency": "#A1794A",
}

AUGMENT_LINESTYLES = {
    "none": "-",
    "spatial": "--",
    "frequency": "-.",
    "spatial_frequency": ":",
}


def _choose_train_fn(config: dict[str, Any], pipeline: str):
    patching_enabled = bool(config.get("patching", {}).get("enabled", False))
    use_patch_pipeline = pipeline == "patch" or (pipeline == "auto" and patching_enabled)
    train_fn = train_and_evaluate_v2 if use_patch_pipeline else train_and_evaluate
    return train_fn, use_patch_pipeline


def _build_variant_config(
    base_config: dict[str, Any],
    *,
    variant_name: str,
    seed: int,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = copy.deepcopy(base_config)
    updated["seed"] = seed
    updated["paths"]["output_root"] = str(output_root)
    if updated.get("patching", {}).get("cache_enabled", False):
        updated.setdefault("patching", {})["cache_dir"] = str(output_root / "patch_cache")

    variant = AUGMENT_VARIANTS[variant_name]
    updated.setdefault("augment", {})["mode"] = variant["mode"]
    if variant["methods"] is None:
        updated["augment"].pop("methods", None)
    else:
        updated["augment"]["methods"] = list(variant["methods"])

    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        updated["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        updated["training"]["device"] = args.device
    if args.learning_rate is not None:
        updated["training"]["learning_rate"] = args.learning_rate
    if args.deterministic:
        updated.setdefault("reproducibility", {})["deterministic"] = True

    return updated


def _load_predictions(predictions_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.array([float(row["anomaly_score"]) for row in rows], dtype=np.float32)
    return labels, scores


def _build_roc_statistics(prediction_paths: list[Path]) -> dict[str, Any]:
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    tpr_curves: list[np.ndarray] = []
    aucs: list[float] = []
    for predictions_path in prediction_paths:
        labels, scores = _load_predictions(predictions_path)
        fpr, tpr, _ = roc_curve(labels, scores)
        unique_fpr = np.unique(fpr)
        unique_tpr = np.array([float(np.max(tpr[fpr == value])) for value in unique_fpr], dtype=np.float32)
        if unique_fpr[0] > 0.0:
            unique_fpr = np.insert(unique_fpr, 0, 0.0)
            unique_tpr = np.insert(unique_tpr, 0, 0.0)
        if unique_fpr[-1] < 1.0:
            unique_fpr = np.append(unique_fpr, 1.0)
            unique_tpr = np.append(unique_tpr, 1.0)
        interpolator = PchipInterpolator(unique_fpr, unique_tpr)
        interpolated = interpolator(fpr_grid)
        interpolated = np.maximum.accumulate(np.clip(interpolated, 0.0, 1.0))
        interpolated[0] = 0.0
        interpolated[-1] = 1.0
        tpr_curves.append(interpolated.astype(np.float32))

        aucs.append(float(roc_auc_score(labels, scores)))

    tpr_array = np.stack(tpr_curves, axis=0)
    return {
        "fpr_grid": fpr_grid.tolist(),
        "mean_tpr": np.mean(tpr_array, axis=0).tolist(),
        "std_tpr": np.std(tpr_array, axis=0).tolist(),
        "auc_mean": float(fmean(aucs)),
        "auc_std": float(pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "num_runs": len(aucs),
    }


def _plot_roc_with_shading(roc_stats: dict[str, dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(8.5, 6.5))
    colors = {
        "none": "#2f4858",
        "spatial": "#d1495b",
        "frequency": "#2e933c",
        "spatial_frequency": "#edae49",
    }

    for variant_name, stats in roc_stats.items():
        fpr_grid = np.array(stats["fpr_grid"], dtype=np.float32)
        mean_tpr = np.array(stats["mean_tpr"], dtype=np.float32)
        std_tpr = np.array(stats["std_tpr"], dtype=np.float32)
        color = colors.get(variant_name, None)
        label = AUGMENT_VARIANTS[variant_name]["label"]
        auc_mean = float(stats["auc_mean"])
        auc_std = float(stats["auc_std"])

        plt.plot(
            fpr_grid,
            mean_tpr,
            linewidth=2.2,
            color=color,
            label=f"{label} (AUC={auc_mean:.4f}±{auc_std:.4f})",
        )
        plt.fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0.0, 1.0),
            np.clip(mean_tpr + std_tpr, 0.0, 1.0),
            color=color,
            alpha=0.16,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Augmentation ROC Comparison (Mean ± Std)")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter 3 augmentation study with multiseed ROC shading")
    parser.add_argument("--config", default="configs/chapter3_luae.yaml", help="Path to base chapter 3 config")
    parser.add_argument("--variants", nargs="+", choices=list(AUGMENT_VARIANTS.keys()), default=list(AUGMENT_VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--pipeline", choices=("auto", "base", "patch"), default="auto", help="Pipeline to use")
    parser.add_argument("--output-root", default="D:/pythonProject2/outputs/chapter3_augmentation_study", help="Output root")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for quick checks")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have metrics")
    args = parser.parse_args()

    base_config = load_yaml(args.config)
    output_root = ensure_dir(Path(args.output_root))

    result_rows: list[dict[str, Any]] = []
    roc_stats: dict[str, dict[str, Any]] = {}

    for variant_name in args.variants:
        prediction_paths: list[Path] = []
        variant_root = ensure_dir(output_root / variant_name)
        for seed in args.seeds:
            run_root = ensure_dir(variant_root / f"seed_{seed}")
            metrics_path = run_root / "metrics" / "test_metrics.json"
            predictions_path = run_root / "metrics" / "test_predictions.csv"
            if args.skip_existing and metrics_path.exists() and predictions_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                config = _build_variant_config(
                    base_config,
                    variant_name=variant_name,
                    seed=seed,
                    output_root=run_root,
                    args=args,
                )
                train_fn, use_patch_pipeline = _choose_train_fn(config, args.pipeline)
                result = train_fn(config, max_test_samples=args.max_test_samples)
                metrics = result["metrics"]
                metrics["pipeline"] = "patch" if use_patch_pipeline else "base"
            prediction_paths.append(predictions_path)

            row = {
                "variant": variant_name,
                "variant_label": AUGMENT_VARIANTS[variant_name]["label"],
                "seed": int(seed),
            }
            row.update(metrics)
            result_rows.append(row)
            print(
                f"variant={variant_name} | seed={seed} | "
                f"AUC={float(metrics['auc']):.4f} | F1={float(metrics['f1_score']):.4f} | "
                f"Recall={float(metrics['recall']):.4f}"
            )

        roc_stats[variant_name] = _build_roc_statistics(prediction_paths)

    summary_rows: list[dict[str, Any]] = []
    for variant_name in args.variants:
        variant_rows = [row for row in result_rows if row["variant"] == variant_name]
        summary = {
            "variant": variant_name,
            "variant_label": AUGMENT_VARIANTS[variant_name]["label"],
            "num_runs": len(variant_rows),
        }
        for metric_key in ("accuracy", "precision", "recall", "f1_score", "auc"):
            values = [float(row[metric_key]) for row in variant_rows]
            summary[f"{metric_key}_mean"] = float(fmean(values))
            summary[f"{metric_key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        summary["roc_auc_mean"] = float(roc_stats[variant_name]["auc_mean"])
        summary["roc_auc_std"] = float(roc_stats[variant_name]["auc_std"])
        summary_rows.append(summary)

    save_csv(result_rows, output_root / "augmentation_multiseed_results.csv")
    save_csv(summary_rows, output_root / "augmentation_summary.csv")
    save_json({"roc_stats": roc_stats}, output_root / "augmentation_roc_stats.json")
    _plot_roc_with_shading(roc_stats, output_root / "augmentation_roc_mean_std.png")

    print(f"summary_csv={output_root / 'augmentation_summary.csv'}")
    print(f"roc_plot={output_root / 'augmentation_roc_mean_std.png'}")
    print(f"roc_stats_json={output_root / 'augmentation_roc_stats.json'}")


if __name__ == "__main__":
    main()
