from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_history(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["history"]


def _plot_training_curve(history: dict, path: Path, *, language: str) -> None:
    _configure_plot_style()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8.8, 5.8))
    plt.plot(epochs, history["train_loss"], color="#355C7D", linewidth=2.2, linestyle="-", label="训练损失" if language == "zh" else "Train Loss")
    plt.plot(epochs, history["val_loss"], color="#A86F3D", linewidth=2.2, linestyle="--", label="验证损失" if language == "zh" else "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("损失值" if language == "zh" else "Loss")
    plt.title("LUAE训练曲线" if language == "zh" else "LUAE Training Curve")
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render bilingual chapter 3 training curves")
    parser.add_argument("--history-json", default="D:/pythonProject2/outputs/chapter3_fullcurve_run/metrics/training_history.json")
    parser.add_argument("--output-dir", default="D:/pythonProject2/outputs/chapter3_fullcurve_run/paper_assets")
    args = parser.parse_args()

    history = _load_history(Path(args.history_json))
    output_dir = Path(args.output_dir)
    _plot_training_curve(history, output_dir / "chapter3_training_curve_zh.png", language="zh")
    _plot_training_curve(history, output_dir / "chapter3_training_curve_en.png", language="en")
    print(f"curve_zh={output_dir / 'chapter3_training_curve_zh.png'}")
    print(f"curve_en={output_dir / 'chapter3_training_curve_en.png'}")


if __name__ == "__main__":
    main()
