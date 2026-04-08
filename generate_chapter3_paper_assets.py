from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from torch import amp

from spray_defect.anomaly import compute_batch_scores
from spray_defect.config import choose_device, ensure_dir
from spray_defect.data_v2 import build_dataloaders_v2
from spray_defect.models import LightweightUNetAutoEncoder


METRIC_FIELDS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1_score"),
    ("AUC", "auc"),
]

METHOD_COLORS = {
    "AE": "#7A7A7A",
    "PaDiM": "#5E7D6A",
    "ResNet18": "#A86F3D",
    "LUAE (Ours)": "#355C7D",
}

METHOD_LINESTYLES = {
    "AE": "-",
    "PaDiM": "--",
    "ResNet18": "-.",
    "LUAE (Ours)": ":",
}


def _configure_plot_style() -> None:
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_best_ae_metrics(outputs_root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for metrics_path in outputs_root.rglob("test_metrics.json"):
        run_name = metrics_path.parent.parent.name.lower()
        if "baseline" not in run_name:
            continue
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue
        auc = _to_float(metrics.get("auc"))
        if auc is None or np.isnan(auc):
            continue
        candidates.append((auc, metrics_path))

    if not candidates:
        raise FileNotFoundError("No baseline AE metrics found under outputs.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _build_default_comparison_rows(outputs_root: Path, luae_metrics_path: Path) -> list[dict[str, Any]]:
    ae_metrics_path = _find_best_ae_metrics(outputs_root)
    ae_metrics = _load_json(ae_metrics_path)
    luae_metrics = _load_json(luae_metrics_path)

    rows = []
    for method_name, metrics, source in [
        ("AE", ae_metrics, str(ae_metrics_path)),
        ("PaDiM", {}, ""),
        ("ResNet18", {}, ""),
        ("LUAE (Ours)", luae_metrics, str(luae_metrics_path)),
    ]:
        row: dict[str, Any] = {"Method": method_name, "Source": source}
        for _, key in METRIC_FIELDS:
            value = metrics.get(key)
            row[key] = "" if value is None else value
        rows.append(row)
    return rows


def _apply_metrics_to_rows(rows: list[dict[str, Any]], method_name: str, metrics_path: Path) -> None:
    metrics = _load_json(metrics_path)
    for row in rows:
        if row["Method"] != method_name:
            continue
        row["Source"] = str(metrics_path)
        for _, key in METRIC_FIELDS:
            row[key] = metrics.get(key, "")
        return

    new_row: dict[str, Any] = {"Method": method_name, "Source": str(metrics_path)}
    for _, key in METRIC_FIELDS:
        new_row[key] = metrics.get(key, "")
    rows.append(new_row)


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = ["Method"] + [key for _, key in METRIC_FIELDS] + ["Source"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_comparison_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    headers = ["算法模型"] + [label for label, _ in METRIC_FIELDS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row["Method"]]
        for _, key in METRIC_FIELDS:
            numeric = _to_float(row.get(key))
            values.append("待补充" if numeric is None else f"{numeric:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_table_en(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    headers = ["Method"] + [label for label, _ in METRIC_FIELDS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row["Method"]]
        for _, key in METRIC_FIELDS:
            numeric = _to_float(row.get(key))
            values.append("TBD" if numeric is None else f"{numeric:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_bar_chart(rows: list[dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    valid_rows = [row for row in rows if any(_to_float(row.get(key)) is not None for _, key in METRIC_FIELDS)]
    methods = [row["Method"] for row in valid_rows]
    metrics = [label for label, _ in METRIC_FIELDS]
    if language == "zh":
        metrics = ["准确率", "精确率", "召回率", "F1值", "AUC"]
    values = np.array(
        [
            [_to_float(row.get(key)) if _to_float(row.get(key)) is not None else np.nan for _, key in METRIC_FIELDS]
            for row in valid_rows
        ],
        dtype=np.float32,
    )

    plt.figure(figsize=(10.5, 6.0))
    x = np.arange(len(metrics))
    width = 0.18 if len(methods) >= 4 else 0.24
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for index, method in enumerate(methods):
        bars = plt.bar(
            x + offsets[index],
            values[index],
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#666666"),
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar in bars:
            height = float(bar.get_height())
            if np.isnan(height):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.05)
    plt.ylabel("指标值" if language == "zh" else "Score")
    plt.title("第三章算法定量对比" if language == "zh" else "Chapter 3 Quantitative Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def _plot_line_chart(rows: list[dict[str, Any]], path: Path, *, language: str) -> None:
    _configure_plot_style()
    valid_rows = [row for row in rows if any(_to_float(row.get(key)) is not None for _, key in METRIC_FIELDS)]
    metrics = [label for label, _ in METRIC_FIELDS]
    if language == "zh":
        metrics = ["准确率", "精确率", "召回率", "F1值", "AUC"]
    x = np.arange(len(metrics))

    plt.figure(figsize=(10.5, 6.0))
    for row in valid_rows:
        y = [_to_float(row.get(key)) if _to_float(row.get(key)) is not None else np.nan for _, key in METRIC_FIELDS]
        plt.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            linestyle=METHOD_LINESTYLES.get(row["Method"], "-"),
            color=METHOD_COLORS.get(row["Method"], "#666666"),
            label=row["Method"],
        )
        for index, value in enumerate(y):
            if value is None or np.isnan(value):
                continue
            plt.text(x[index], float(value) + 0.016, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.05)
    plt.ylabel("指标值" if language == "zh" else "Score")
    plt.title("第三章算法指标趋势对比" if language == "zh" else "Chapter 3 Metric Trend Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def _load_luae_model(checkpoint_path: Path, device: torch.device) -> tuple[LightweightUNetAutoEncoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = config.get("model", {})
    model = LightweightUNetAutoEncoder(
        base_channels=int(model_config.get("base_channels", 32)),
        norm_type=str(model_config.get("norm_type", "batchnorm")),
        group_count=int(model_config.get("group_count", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def _select_residual_targets(predictions_path: Path, num_normal: int, num_defect: int) -> list[dict[str, Any]]:
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    normal_rows = [row for row in rows if int(row["label"]) == 0]
    defect_rows = [row for row in rows if int(row["label"]) == 1]
    normal_rows.sort(key=lambda row: float(row["anomaly_score"]))
    defect_rows.sort(key=lambda row: float(row["anomaly_score"]), reverse=True)
    return normal_rows[:num_normal] + defect_rows[:num_defect]


def _collect_residual_examples(
    checkpoint_path: Path,
    predictions_path: Path,
    device_name: str,
    num_normal: int,
    num_defect: int,
) -> list[dict[str, Any]]:
    device = choose_device(device_name)
    model, config = _load_luae_model(checkpoint_path, device)
    loader = build_dataloaders_v2(config)["test"]
    amp_enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
    targets = _select_residual_targets(predictions_path, num_normal=num_normal, num_defect=num_defect)
    target_paths = {row["path"]: row for row in targets}
    best_examples: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            with amp.autocast(device_type=device.type, enabled=amp_enabled):
                reconstructions = model(images)
            _, _, _, _, _, anomaly_maps, _ = compute_batch_scores(images.float(), reconstructions.float(), config["scoring"])

            image_array = images.detach().cpu().numpy()
            recon_array = reconstructions.detach().cpu().numpy()
            for index, path in enumerate(batch["path"]):
                if path not in target_paths:
                    continue
                info = target_paths[path]
                patch_score = float(np.quantile(anomaly_maps[index], 0.995))
                current = best_examples.get(path)
                if current is None or patch_score > float(current["patch_score"]):
                    best_examples[path] = {
                        "path": path,
                        "label": int(info["label"]),
                        "predicted_label": int(info["predicted_label"]),
                        "image_score": float(info["anomaly_score"]),
                        "threshold": float(info.get("threshold", 0.0)),
                        "patch_score": patch_score,
                        "input": image_array[index, 0],
                        "reconstruction": recon_array[index, 0],
                        "heatmap": anomaly_maps[index],
                    }
            if len(best_examples) >= len(target_paths):
                break

    examples = [best_examples[path] for path in target_paths if path in best_examples]
    examples.sort(key=lambda item: (item["label"], -item["image_score"]))
    return examples


def _save_residual_grid(samples: list[dict[str, Any]], path: Path, *, language: str) -> None:
    if not samples:
        return

    _configure_plot_style()
    ensure_dir(path.parent)
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for index, sample in enumerate(samples):
        label_name = ("缺陷" if sample["label"] == 1 else "正常") if language == "zh" else ("Defect" if sample["label"] == 1 else "Normal")
        pred_name = ("缺陷" if sample["predicted_label"] == 1 else "正常") if language == "zh" else ("Defect" if sample["predicted_label"] == 1 else "Normal")
        threshold = max(float(sample.get("threshold", 0.0)), 1e-8)
        score_ratio = float(sample["image_score"]) / threshold
        axes[index, 0].imshow(sample["input"], cmap="gray")
        if language == "zh":
            axes[index, 0].set_title(
                f"输入Patch\n{label_name} | 图像分数={sample['image_score']:.4f} | 阈值比={score_ratio:.2f}x"
            )
        else:
            axes[index, 0].set_title(
                f"Input Patch\n{label_name} | img={sample['image_score']:.4f} | ratio={score_ratio:.2f}x"
            )
        axes[index, 1].imshow(sample["reconstruction"], cmap="gray")
        axes[index, 1].set_title("重建结果" if language == "zh" else "Reconstruction")
        axes[index, 2].imshow(sample["heatmap"], cmap="inferno")
        axes[index, 2].set_title(("残差热图\n预测=" if language == "zh" else "Residual Map\nPred=") + pred_name)
        for axis in axes[index]:
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chapter 3 paper-ready comparison assets")
    parser.add_argument("--outputs-root", default="D:/pythonProject2/outputs", help="Root outputs directory")
    parser.add_argument("--luae-run", default="chapter3_repro", help="LUAE output directory name")
    parser.add_argument("--comparison-csv", default=None, help="Optional comparison CSV path")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated assets")
    parser.add_argument("--device", default="auto", help="Device for residual visualization")
    parser.add_argument("--num-normal", type=int, default=2, help="Number of normal residual examples")
    parser.add_argument("--num-defect", type=int, default=4, help="Number of defect residual examples")
    parser.add_argument("--padim-metrics", default=None, help="Optional PaDiM metrics JSON path")
    parser.add_argument("--resnet18-metrics", default=None, help="Optional ResNet18 metrics JSON path")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    luae_dir = outputs_root / args.luae_run
    luae_metrics_path = luae_dir / "metrics" / "test_metrics.json"
    luae_predictions_path = luae_dir / "metrics" / "test_predictions.csv"
    luae_checkpoint_path = luae_dir / "checkpoints" / "best_model.pt"
    output_dir = Path(args.output_dir) if args.output_dir else ensure_dir(luae_dir / "paper_assets")
    ensure_dir(output_dir)

    comparison_csv_path = Path(args.comparison_csv) if args.comparison_csv else output_dir / "chapter3_method_comparison.csv"
    if comparison_csv_path.exists():
        rows = _load_comparison_csv(comparison_csv_path)
    else:
        rows = _build_default_comparison_rows(outputs_root, luae_metrics_path)

    if args.padim_metrics:
        _apply_metrics_to_rows(rows, "PaDiM", Path(args.padim_metrics))
    if args.resnet18_metrics:
        _apply_metrics_to_rows(rows, "ResNet18", Path(args.resnet18_metrics))

    _apply_metrics_to_rows(rows, "LUAE (Ours)", luae_metrics_path)
    _write_comparison_csv(rows, comparison_csv_path)

    _write_markdown_table(rows, output_dir / "chapter3_method_comparison_zh.md")
    _write_markdown_table_en(rows, output_dir / "chapter3_method_comparison_en.md")
    _plot_bar_chart(rows, output_dir / "chapter3_method_comparison_bar_zh.png", language="zh")
    _plot_bar_chart(rows, output_dir / "chapter3_method_comparison_bar_en.png", language="en")
    _plot_line_chart(rows, output_dir / "chapter3_method_comparison_line_zh.png", language="zh")
    _plot_line_chart(rows, output_dir / "chapter3_method_comparison_line_en.png", language="en")
    _write_comparison_csv(rows, comparison_csv_path)

    residual_examples = _collect_residual_examples(
        luae_checkpoint_path,
        luae_predictions_path,
        device_name=args.device,
        num_normal=args.num_normal,
        num_defect=args.num_defect,
    )
    _save_residual_grid(residual_examples, output_dir / "chapter3_luae_residual_maps_zh.png", language="zh")
    _save_residual_grid(residual_examples, output_dir / "chapter3_luae_residual_maps_en.png", language="en")

    print(f"comparison_csv={comparison_csv_path}")
    print(f"table_md_zh={output_dir / 'chapter3_method_comparison_zh.md'}")
    print(f"table_md_en={output_dir / 'chapter3_method_comparison_en.md'}")
    print(f"bar_chart_zh={output_dir / 'chapter3_method_comparison_bar_zh.png'}")
    print(f"bar_chart_en={output_dir / 'chapter3_method_comparison_bar_en.png'}")
    print(f"line_chart_zh={output_dir / 'chapter3_method_comparison_line_zh.png'}")
    print(f"line_chart_en={output_dir / 'chapter3_method_comparison_line_en.png'}")
    print(f"residual_maps_zh={output_dir / 'chapter3_luae_residual_maps_zh.png'}")
    print(f"residual_maps_en={output_dir / 'chapter3_luae_residual_maps_en.png'}")


if __name__ == "__main__":
    main()
