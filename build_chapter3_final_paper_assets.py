from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import roc_auc_score, roc_curve

import generate_chapter3_paper_assets as chapter3_assets


METHOD_ORDER = ["AE", "PaDiM", "ResNet18", "LUAE"]

METHOD_LABELS = {
    "AE": {"zh": "AE", "en": "AE"},
    "PaDiM": {"zh": "PaDiM", "en": "PaDiM"},
    "ResNet18": {"zh": "ResNet18", "en": "ResNet18"},
    "LUAE": {"zh": "LUAE（本文算法）", "en": "LUAE (This Work)"},
}

METHOD_COLORS = {
    "AE": "#7A7A7A",
    "PaDiM": "#5E7D6A",
    "ResNet18": "#A86F3D",
    "LUAE": "#274C77",
}

METHOD_LINESTYLES = {
    "AE": "-",
    "PaDiM": "--",
    "ResNet18": "-.",
    "LUAE": "-",
}

METRICS = [
    ("accuracy", {"zh": "准确率", "en": "Accuracy"}),
    ("precision", {"zh": "精确率", "en": "Precision"}),
    ("recall", {"zh": "召回率", "en": "Recall"}),
    ("f1_score", {"zh": "F1值", "en": "F1-score"}),
    ("auc", {"zh": "AUC", "en": "AUC"}),
]

AUGMENT_VARIANTS = {
    "none": {"zh": "无增强", "en": "No Augmentation"},
    "spatial": {"zh": "空间增强", "en": "Spatial"},
    "frequency": {"zh": "频率增强", "en": "Frequency"},
    "spatial_frequency": {"zh": "空间+频率增强", "en": "Spatial + Frequency"},
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
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_comparison_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_key in METHOD_ORDER:
        metrics = _load_json(paths[method_key])
        row = {"method_key": method_key}
        for metric_key, _ in METRICS:
            row[metric_key] = float(metrics[metric_key])
        rows.append(row)
    return rows


def _best_metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric_key: max(float(row[metric_key]) for row in rows) for metric_key, _ in METRICS}


def _write_table_markdown(rows: list[dict[str, Any]], path: Path, *, language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["算法模型" if language == "zh" else "Method"] + [labels[language] for _, labels in METRICS]
    best_values = _best_metric_values(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [METHOD_LABELS[row["method_key"]][language]]
        for metric_key, _ in METRICS:
            value = float(row[metric_key])
            text = f"{value:.4f}"
            if np.isclose(value, best_values[metric_key]):
                text = f"**{text}**"
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Method"] + [labels["en"] for _, labels in METRICS]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Method": METHOD_LABELS[row["method_key"]]["en"],
                    **{labels["en"]: f"{float(row[key]):.4f}" for key, labels in METRICS},
                }
            )


def _plot_bar(rows: list[dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    metrics = [labels[language] for _, labels in METRICS]
    values = np.array([[float(row[key]) for key, _ in METRICS] for row in rows], dtype=np.float32)
    methods = [row["method_key"] for row in rows]
    best_values = _best_metric_values(rows)

    plt.figure(figsize=(10.6, 6.2))
    x = np.arange(len(metrics))
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for method_index, method_key in enumerate(methods):
        bars = plt.bar(
            x + offsets[method_index],
            values[method_index],
            width=width,
            color=METHOD_COLORS[method_key],
            edgecolor="#2C2C2C",
            linewidth=0.6,
            label=METHOD_LABELS[method_key][language],
            alpha=1.0 if method_key == "LUAE" else 0.9,
        )
        for metric_index, bar in enumerate(bars):
            metric_key = METRICS[metric_index][0]
            height = float(bar.get_height())
            is_best = np.isclose(height, best_values[metric_key])
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                fontweight="bold" if is_best else "normal",
                color="#111111" if is_best else "#333333",
            )

    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.05)
    plt.ylabel("指标值" if language == "zh" else "Score")
    plt.grid(axis="y", alpha=0.22)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _plot_line(rows: list[dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    metrics = [labels[language] for _, labels in METRICS]
    x = np.arange(len(metrics))
    best_values = _best_metric_values(rows)
    y_offsets = {
        "AE": -0.030,
        "PaDiM": -0.010,
        "ResNet18": 0.012,
        "LUAE": 0.032,
    }

    plt.figure(figsize=(10.8, 6.2))
    for row in rows:
        method_key = row["method_key"]
        y = np.array([float(row[key]) for key, _ in METRICS], dtype=np.float32)
        plt.plot(
            x,
            y,
            color=METHOD_COLORS[method_key],
            linestyle=METHOD_LINESTYLES[method_key],
            linewidth=2.6 if method_key == "LUAE" else 2.0,
            marker="o",
            markersize=6 if method_key == "LUAE" else 5,
            label=METHOD_LABELS[method_key][language],
            zorder=3 if method_key == "LUAE" else 2,
        )
        for index, value in enumerate(y):
            metric_key = METRICS[index][0]
            is_best = np.isclose(float(value), best_values[metric_key])
            plt.text(
                x[index],
                float(value) + y_offsets[method_key],
                f"{float(value):.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if is_best else "normal",
                color="#111111" if is_best else "#333333",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.05)
    plt.ylabel("指标值" if language == "zh" else "Score")
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _plot_training_curve(history_path: Path, path: Path, *, language: str) -> None:
    _configure_plot_style()
    history = _load_json(history_path)["history"]
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8.8, 5.8))
    plt.plot(
        epochs,
        history["train_loss"],
        color="#274C77",
        linewidth=2.3,
        linestyle="-",
        label="训练损失" if language == "zh" else "Train Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        color="#A86F3D",
        linewidth=2.1,
        linestyle="--",
        label="验证损失" if language == "zh" else "Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("损失值" if language == "zh" else "Loss")
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


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


def _build_augmentation_stats(study_root: Path) -> dict[str, dict[str, Any]]:
    roc_stats: dict[str, dict[str, Any]] = {}
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    for variant_key in AUGMENT_VARIANTS:
        prediction_paths = sorted((study_root / variant_key).glob("seed_*/metrics/test_predictions.csv"))
        tpr_curves: list[np.ndarray] = []
        aucs: list[float] = []
        for prediction_path in prediction_paths:
            labels, scores = _load_predictions(prediction_path)
            tpr_curves.append(_smooth_single_roc(labels, scores, fpr_grid))
            aucs.append(float(roc_auc_score(labels, scores)))
        tpr_array = np.stack(tpr_curves, axis=0)
        roc_stats[variant_key] = {
            "fpr_grid": fpr_grid.tolist(),
            "mean_tpr": np.mean(tpr_array, axis=0).tolist(),
            "std_tpr": np.std(tpr_array, axis=0).tolist(),
            "auc_mean": float(fmean(aucs)),
            "auc_std": float(pstdev(aucs)) if len(aucs) > 1 else 0.0,
            "num_runs": len(aucs),
        }
    return roc_stats


def _plot_augmentation_roc(roc_stats: dict[str, dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    plt.figure(figsize=(8.8, 6.6))
    for variant_key, stats in roc_stats.items():
        fpr_grid = np.array(stats["fpr_grid"], dtype=np.float32)
        mean_tpr = np.array(stats["mean_tpr"], dtype=np.float32)
        std_tpr = np.array(stats["std_tpr"], dtype=np.float32)
        plt.plot(
            fpr_grid,
            mean_tpr,
            color=AUGMENT_COLORS[variant_key],
            linestyle=AUGMENT_LINESTYLES[variant_key],
            linewidth=2.4,
            label=f"{AUGMENT_VARIANTS[variant_key][language]} (AUC={float(stats['auc_mean']):.4f}±{float(stats['auc_std']):.4f})",
        )
        plt.fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0.0, 1.0),
            np.clip(mean_tpr + std_tpr, 0.0, 1.0),
            color=AUGMENT_COLORS[variant_key],
            alpha=0.13,
            linewidth=0.0,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("假阳性率 FPR" if language == "zh" else "False Positive Rate")
    plt.ylabel("真阳性率 TPR" if language == "zh" else "True Positive Rate")
    plt.grid(alpha=0.22)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _write_augmentation_table(roc_stats: dict[str, dict[str, Any]], path: Path, *, language: str) -> None:
    headers = ["增强方式" if language == "zh" else "Augmentation", "AUC Mean", "AUC Std", "n"]
    best_auc = max(float(stats["auc_mean"]) for stats in roc_stats.values())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for variant_key in AUGMENT_VARIANTS:
        auc_mean = float(roc_stats[variant_key]["auc_mean"])
        auc_std = float(roc_stats[variant_key]["auc_std"])
        auc_text = f"{auc_mean:.4f}"
        if np.isclose(auc_mean, best_auc):
            auc_text = f"**{auc_text}**"
        lines.append(
            "| "
            + " | ".join(
                [
                    AUGMENT_VARIANTS[variant_key][language],
                    auc_text,
                    f"{auc_std:.4f}",
                    str(int(roc_stats[variant_key]["num_runs"])),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_residual_grid(checkpoint_path: Path, predictions_path: Path, path: Path, *, language: str) -> None:
    samples = chapter3_assets._collect_residual_examples(
        checkpoint_path,
        predictions_path,
        device_name="auto",
        num_normal=2,
        num_defect=4,
    )
    chapter3_assets._save_residual_grid(samples, path, language=language)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final chapter 3 paper assets with unified naming")
    parser.add_argument("--outputs-root", default="D:/pythonProject2/outputs")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    final_dir = outputs_root / "00_论文总目录_最终版" / "03_第三章论文输出" / "99_最终投稿版"
    final_dir.mkdir(parents=True, exist_ok=True)

    comparison_paths = {
        "AE": outputs_root / "baseline_p768_b32_lr5e4_cosine" / "metrics" / "test_metrics.json",
        "PaDiM": outputs_root / "chapter3_feature_benchmarks_final" / "padim" / "metrics" / "test_metrics.json",
        "ResNet18": outputs_root / "chapter3_feature_benchmarks_final" / "resnet18" / "metrics" / "test_metrics.json",
        "LUAE": outputs_root / "chapter3_repro" / "metrics" / "test_metrics.json",
    }
    comparison_rows = _load_comparison_rows(comparison_paths)

    _write_table_markdown(comparison_rows, final_dir / "Table3-1_四种算法定量对比表_zh.md", language="zh")
    _write_table_markdown(comparison_rows, final_dir / "Table3-1_Quantitative_Comparison_of_Four_Algorithms_en.md", language="en")
    _write_table_csv(comparison_rows, final_dir / "Table3-1_四种算法定量对比表.csv")
    _plot_training_curve(
        outputs_root / "chapter3_fullcurve_run" / "metrics" / "training_history.json",
        final_dir / "Fig3-1_LUAE训练曲线_zh.png",
        language="zh",
    )
    _plot_training_curve(
        outputs_root / "chapter3_fullcurve_run" / "metrics" / "training_history.json",
        final_dir / "Fig3-1_LUAE_Training_Curve_en.png",
        language="en",
    )
    _plot_bar(comparison_rows, final_dir / "Fig3-2_四种算法定量对比柱状图_zh.png", language="zh")
    _plot_bar(comparison_rows, final_dir / "Fig3-2_Quantitative_Bar_Comparison_of_Four_Algorithms_en.png", language="en")
    _plot_line(comparison_rows, final_dir / "Fig3-3_四种算法定量对比折线图_zh.png", language="zh")
    _plot_line(comparison_rows, final_dir / "Fig3-3_Quantitative_Line_Comparison_of_Four_Algorithms_en.png", language="en")
    _render_residual_grid(
        outputs_root / "chapter3_repro" / "checkpoints" / "best_model.pt",
        outputs_root / "chapter3_repro" / "metrics" / "test_predictions.csv",
        final_dir / "Fig3-4_LUAE残差可视化结果_zh.png",
        language="zh",
    )
    _render_residual_grid(
        outputs_root / "chapter3_repro" / "checkpoints" / "best_model.pt",
        outputs_root / "chapter3_repro" / "metrics" / "test_predictions.csv",
        final_dir / "Fig3-4_LUAE_Residual_Visualization_en.png",
        language="en",
    )

    roc_stats = _build_augmentation_stats(outputs_root / "chapter3_augmentation_study")
    (final_dir / "Fig3-5_四种增强方式ROC统计.json").write_text(
        json.dumps({"roc_stats": roc_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_augmentation_table(roc_stats, final_dir / "Table3-2_四种增强方式AUC统计表_zh.md", language="zh")
    _write_augmentation_table(roc_stats, final_dir / "Table3-2_AUC_Statistics_of_Four_Augmentation_Strategies_en.md", language="en")
    _plot_augmentation_roc(roc_stats, final_dir / "Fig3-5_四种增强方式ROC对比图_zh.png", language="zh")
    _plot_augmentation_roc(roc_stats, final_dir / "Fig3-5_ROC_Comparison_of_Four_Augmentation_Strategies_en.png", language="en")

    # Copy notes into final folder for direct citation
    _copy_file(
        outputs_root / "chapter3_augmentation_study" / "paper_assets" / "augmentation_experiment_notes_zh.md",
        final_dir / "附_增强实验说明_zh.md",
    )
    _copy_file(
        outputs_root / "chapter3_augmentation_study" / "paper_assets" / "augmentation_experiment_notes_en.md",
        final_dir / "Appendix_Augmentation_Experiment_Notes_en.md",
    )

    print(f"final_dir={final_dir}")


if __name__ == "__main__":
    main()
