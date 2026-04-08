from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS_ZH = ["AE", "PaDiM", "ResNet18", "LUAE（本文算法）"]
METHODS_EN = ["AE", "PaDiM", "ResNet18", "LUAE (This Work)"]

METRICS = [
    ("accuracy", "准确率", "Accuracy", "#A0A7B4"),
    ("precision", "精确率", "Precision", "#7C8B9A"),
    ("recall", "召回率", "Recall", "#5E7D6A"),
    ("f1_score", "F1值", "F1-score", "#A86F3D"),
    ("auc", "AUC", "AUC", "#274C77"),
]


def _configure_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_metrics() -> dict[str, dict]:
    base = Path(r"D:\pythonProject2\outputs")
    return {
        "AE": json.loads((base / "baseline_p768_b32_lr5e4_cosine" / "metrics" / "test_metrics.json").read_text(encoding="utf-8")),
        "PaDiM": json.loads((base / "chapter3_feature_benchmarks_final" / "padim" / "metrics" / "test_metrics.json").read_text(encoding="utf-8")),
        "ResNet18": json.loads((base / "chapter3_feature_benchmarks_final" / "resnet18" / "metrics" / "test_metrics.json").read_text(encoding="utf-8")),
        "LUAE": json.loads((base / "chapter3_repro" / "metrics" / "test_metrics.json").read_text(encoding="utf-8")),
    }


def _final_dir() -> Path:
    base = Path(r"D:\pythonProject2\outputs")
    final_root = next(path for path in base.iterdir() if path.is_dir() and path.name.startswith("00_"))
    chapter3_root = next(path for path in final_root.iterdir() if path.is_dir() and path.name.startswith("03_"))
    return next(path for path in chapter3_root.iterdir() if path.is_dir() and path.name.startswith("99_"))


def _plot(language: str, output_path: Path) -> None:
    _configure_style()
    metrics = _load_metrics()
    method_keys = ["AE", "PaDiM", "ResNet18", "LUAE"]
    method_labels = METHODS_ZH if language == "zh" else METHODS_EN
    metric_labels = [metric_zh if language == "zh" else metric_en for _, metric_zh, metric_en, _ in METRICS]

    x = np.arange(len(method_keys))
    width = 0.14
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2.0) * width
    best_values = {
        metric_key: max(float(metrics[method_key][metric_key]) for method_key in method_keys)
        for metric_key, _, _, _ in METRICS
    }

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    # Highlight the LUAE group subtly.
    ax.axvspan(x[-1] - 0.42, x[-1] + 0.42, color="#DCE6F2", alpha=0.55, zorder=0)

    for metric_index, (metric_key, _, _, color) in enumerate(METRICS):
        values = [float(metrics[method_key][metric_key]) for method_key in method_keys]
        bars = ax.bar(
            x + offsets[metric_index],
            values,
            width=width,
            color=color,
            edgecolor="#2C2C2C",
            linewidth=0.6,
            label=metric_labels[metric_index],
            zorder=2,
        )
        for bar, value in zip(bars, values):
            is_best = np.isclose(value, best_values[metric_key])
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                fontweight="bold" if is_best else "normal",
                color="#111111" if is_best else "#333333",
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    # Emphasize LUAE tick label.
    xticklabels = ax.get_xticklabels()
    if xticklabels:
        xticklabels[-1].set_fontweight("bold")
        xticklabels[-1].set_color("#1F3F5B")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("指标值" if language == "zh" else "Score")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        ncol=5,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#888888",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def main() -> None:
    target_dir = _final_dir()
    zh_path = target_dir / "Fig3-2b_四种算法模型指标分组柱状图_zh.png"
    en_path = target_dir / "Fig3-2b_Quantitative_Grouped_Bar_Chart_of_Four_Algorithms_en.png"
    _plot("zh", zh_path)
    _plot("en", en_path)

    # Remove older temporary names if they exist to avoid confusion.
    for obsolete in (
        target_dir / "Fig3-2b_GroupedBar_FourAlgorithms_zh.png",
        target_dir / "Fig3-2b_GroupedBar_FourAlgorithms_en.png",
    ):
        if obsolete.exists():
            try:
                obsolete.unlink()
            except PermissionError:
                pass

    print(zh_path)
    print(en_path)


if __name__ == "__main__":
    main()
