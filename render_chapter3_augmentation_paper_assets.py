from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import roc_auc_score, roc_curve


AUGMENT_VARIANTS: dict[str, dict[str, str]] = {
    "none": {"label_en": "No Augmentation", "label_zh": "无增强"},
    "spatial": {"label_en": "Spatial", "label_zh": "空间增强"},
    "frequency": {"label_en": "Frequency", "label_zh": "频率增强"},
    "spatial_frequency": {"label_en": "Spatial + Frequency", "label_zh": "空间+频率增强"},
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


def _configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_predictions(predictions_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.array([float(row["anomaly_score"]) for row in rows], dtype=np.float32)
    return labels, scores


def _smooth_single_roc(labels: np.ndarray, scores: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
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
    smoothed = interpolator(fpr_grid)
    smoothed = np.maximum.accumulate(np.clip(smoothed, 0.0, 1.0))
    smoothed[0] = 0.0
    smoothed[-1] = 1.0
    return smoothed.astype(np.float32)


def _build_variant_roc_stats(variant_root: Path) -> dict[str, Any]:
    prediction_paths = sorted(variant_root.glob("seed_*/metrics/test_predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files found under {variant_root}")

    fpr_grid = np.linspace(0.0, 1.0, 1001)
    tpr_curves: list[np.ndarray] = []
    aucs: list[float] = []
    for prediction_path in prediction_paths:
        labels, scores = _load_predictions(prediction_path)
        tpr_curves.append(_smooth_single_roc(labels, scores, fpr_grid))
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


def _plot_roc_with_shading(roc_stats: dict[str, dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.8, 6.6))

    for variant_name, stats in roc_stats.items():
        fpr_grid = np.array(stats["fpr_grid"], dtype=np.float32)
        mean_tpr = np.array(stats["mean_tpr"], dtype=np.float32)
        std_tpr = np.array(stats["std_tpr"], dtype=np.float32)
        label = AUGMENT_VARIANTS[variant_name]["label_zh" if language == "zh" else "label_en"]
        auc_mean = float(stats["auc_mean"])
        auc_std = float(stats["auc_std"])

        plt.plot(
            fpr_grid,
            mean_tpr,
            color=AUGMENT_COLORS[variant_name],
            linestyle=AUGMENT_LINESTYLES[variant_name],
            linewidth=2.4,
            label=f"{label} (AUC={auc_mean:.4f}±{auc_std:.4f})",
        )
        plt.fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0.0, 1.0),
            np.clip(mean_tpr + std_tpr, 0.0, 1.0),
            color=AUGMENT_COLORS[variant_name],
            alpha=0.13,
            linewidth=0.0,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("假阳性率 FPR" if language == "zh" else "False Positive Rate")
    plt.ylabel("真阳性率 TPR" if language == "zh" else "True Positive Rate")
    plt.title("四种增强方式ROC对比（均值±标准差）" if language == "zh" else "ROC Comparison of Four Augmentation Strategies (Mean ± Std)")
    plt.grid(alpha=0.22)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def _write_summary_csv(roc_stats: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant_name, stats in roc_stats.items():
        rows.append(
            {
                "variant": variant_name,
                "variant_label_en": AUGMENT_VARIANTS[variant_name]["label_en"],
                "variant_label_zh": AUGMENT_VARIANTS[variant_name]["label_zh"],
                "num_runs": int(stats["num_runs"]),
                "auc_mean": float(stats["auc_mean"]),
                "auc_std": float(stats["auc_std"]),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, *, language: str, variants: list[str], seeds: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if language == "zh":
        lines = [
            "# 第三章增强消融实验说明",
            "",
            "1. 任务背景：工业喷涂缺陷二分类任务，类别为正常（normal）与缺陷（defect）。",
            "2. 数据划分：固定使用 `configs/chapter3_split_manifest.json` 的训练/验证/测试划分。",
            "3. 模型基底：本文 LUAE，patch-aware pipeline。",
            f"4. 增强方式：{', '.join(AUGMENT_VARIANTS[name]['label_zh'] for name in variants)}。",
            f"5. 重复实验次数：{len(seeds)} 次独立重复实验，随机种子为 {', '.join(seeds)}。",
            "6. 控制变量：除增强方式外，其余网络结构、数据划分、损失函数、阈值估计方法保持一致。",
            "7. 曲线绘制：每次实验先基于原始 ROC 计算，再使用保形单调插值平滑到统一 FPR 网格，最终计算均值曲线与标准差阴影。",
        ]
    else:
        lines = [
            "# Chapter 3 Augmentation Study Notes",
            "",
            "1. Task background: industrial spray-defect binary classification with normal and defect samples.",
            "2. Dataset split: fixed train/validation/test split defined in `configs/chapter3_split_manifest.json`.",
            "3. Backbone model: the proposed LUAE using the patch-aware pipeline.",
            f"4. Compared augmentations: {', '.join(AUGMENT_VARIANTS[name]['label_en'] for name in variants)}.",
            f"5. Repeated experiments: {len(seeds)} independent runs with seeds {', '.join(seeds)}.",
            "6. Controlled variables: all settings remain unchanged except the augmentation strategy.",
            "7. ROC rendering: each run is first converted to the raw ROC curve, then smoothed with monotonic shape-preserving interpolation on a shared FPR grid before computing mean and standard deviation.",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication-ready augmentation ROC assets from existing runs")
    parser.add_argument("--study-root", default="D:/pythonProject2/outputs/chapter3_augmentation_study", help="Root directory of augmentation study runs")
    parser.add_argument("--output-dir", default=None, help="Output directory for paper-ready augmentation assets")
    parser.add_argument("--variants", nargs="+", choices=list(AUGMENT_VARIANTS.keys()), default=list(AUGMENT_VARIANTS.keys()))
    args = parser.parse_args()

    study_root = Path(args.study_root)
    output_dir = Path(args.output_dir) if args.output_dir else study_root / "paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_stats: dict[str, dict[str, Any]] = {}
    all_seeds: set[str] = set()
    for variant_name in args.variants:
        variant_root = study_root / variant_name
        roc_stats[variant_name] = _build_variant_roc_stats(variant_root)
        all_seeds.update(path.parent.parent.name.replace("seed_", "") for path in variant_root.glob("seed_*/metrics/test_predictions.csv"))

    (output_dir / "augmentation_roc_stats.json").write_text(json.dumps({"roc_stats": roc_stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(roc_stats, output_dir / "augmentation_auc_summary.csv")
    _plot_roc_with_shading(roc_stats, output_dir / "augmentation_roc_mean_std_zh.png", language="zh")
    _plot_roc_with_shading(roc_stats, output_dir / "augmentation_roc_mean_std_en.png", language="en")
    _write_notes(output_dir / "augmentation_experiment_notes_zh.md", language="zh", variants=args.variants, seeds=sorted(all_seeds))
    _write_notes(output_dir / "augmentation_experiment_notes_en.md", language="en", variants=args.variants, seeds=sorted(all_seeds))

    print(f"roc_plot_zh={output_dir / 'augmentation_roc_mean_std_zh.png'}")
    print(f"roc_plot_en={output_dir / 'augmentation_roc_mean_std_en.png'}")
    print(f"summary_csv={output_dir / 'augmentation_auc_summary.csv'}")
    print(f"notes_zh={output_dir / 'augmentation_experiment_notes_zh.md'}")
    print(f"notes_en={output_dir / 'augmentation_experiment_notes_en.md'}")


if __name__ == "__main__":
    main()
