from __future__ import annotations

import argparse
import copy
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from spray_defect.config import ensure_dir, load_yaml, save_csv, save_json
from spray_defect.trainer import train_and_evaluate
from spray_defect.trainer_v2 import train_and_evaluate_v2


def _choose_train_fn(config: dict[str, Any], pipeline: str):
    patching_enabled = bool(config.get("patching", {}).get("enabled", False))
    use_patch_pipeline = pipeline == "patch" or (pipeline == "auto" and patching_enabled)
    train_fn = train_and_evaluate_v2 if use_patch_pipeline else train_and_evaluate
    return train_fn, use_patch_pipeline


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace, seed: int, output_root: Path) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated["seed"] = seed
    updated["paths"]["output_root"] = str(output_root)
    if updated.get("patching", {}).get("cache_enabled", False):
        updated.setdefault("patching", {})["cache_dir"] = str(output_root / "patch_cache")

    if args.epochs is not None:
        updated["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        updated["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        updated["training"]["device"] = args.device
    if args.learning_rate is not None:
        updated["training"]["learning_rate"] = args.learning_rate
    if args.scheduler_type is not None:
        updated.setdefault("training", {}).setdefault("scheduler", {})["type"] = args.scheduler_type
    if args.norm_type is not None:
        updated.setdefault("model", {})["norm_type"] = args.norm_type
    if args.group_count is not None:
        updated.setdefault("model", {})["group_count"] = args.group_count
    if args.deterministic:
        updated.setdefault("reproducibility", {})["deterministic"] = True

    return updated


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "threshold",
        "avg_inference_latency_ms",
        "best_monitor_value",
    ]
    summary: dict[str, Any] = {
        "num_runs": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
    }
    if rows:
        best_row = max(rows, key=lambda row: float(row["auc"]))
        summary["best_seed"] = int(best_row["seed"])
        summary["best_auc"] = float(best_row["auc"])
        summary["best_output_root"] = str(best_row["output_root"])

    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        summary[f"{key}_mean"] = float(fmean(values))
        summary[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter 3 training across multiple seeds")
    parser.add_argument("--config", default="configs/chapter3_luae.yaml", help="Path to the chapter 3 config file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size")
    parser.add_argument("--device", default=None, help="Override the device")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the learning rate")
    parser.add_argument("--scheduler-type", default=None, help="Override scheduler type")
    parser.add_argument("--norm-type", default=None, help="Override model normalization type")
    parser.add_argument("--group-count", type=int, default=None, help="Override group count for group normalization")
    parser.add_argument("--pipeline", choices=("auto", "base", "patch"), default="auto", help="Pipeline to use")
    parser.add_argument("--output-root", default=None, help="Root directory for multiseed outputs")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for quick checks")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training behavior")
    args = parser.parse_args()

    base_config = load_yaml(args.config)
    base_output_root = Path(args.output_root) if args.output_root else Path(base_config["paths"]["output_root"] + "_multiseed")
    ensure_dir(base_output_root)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_output_root = ensure_dir(base_output_root / f"seed_{seed}")
        config = _apply_overrides(base_config, args, seed=seed, output_root=run_output_root)
        train_fn, use_patch_pipeline = _choose_train_fn(config, args.pipeline)
        result = train_fn(config, max_test_samples=args.max_test_samples)
        metrics = result["metrics"]
        row = {
            "seed": int(seed),
            "pipeline": "patch" if use_patch_pipeline else "base",
            "output_root": str(run_output_root),
            "checkpoint_path": str(result["checkpoint_path"]),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"seed={seed} | pipeline={row['pipeline']} | "
            f"AUC={metrics['auc']:.4f} | F1={metrics['f1_score']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

    summary = _aggregate(rows)
    save_csv(rows, base_output_root / "multiseed_results.csv")
    save_json(summary, base_output_root / "multiseed_summary.json")

    print(
        f"summary | seeds={summary['seeds']} | "
        f"AUC_mean={summary.get('auc_mean', float('nan')):.4f} | "
        f"AUC_std={summary.get('auc_std', float('nan')):.4f}"
    )
    print(f"summary_path={base_output_root / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
