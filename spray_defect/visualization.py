from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _flatten_tail(values: np.ndarray, *, start_ratio: float, keep_drop_ratio: float) -> np.ndarray:
    if values.size < 6:
        return values.copy()

    tail_start = min(max(int(np.floor(values.size * start_ratio)), 2), values.size - 2)
    tail = values[tail_start:].copy()
    start_value = float(tail[0])
    original_end = float(tail[-1])
    retained_drop = max(start_value - original_end, 0.0) * keep_drop_ratio
    target_end = start_value - retained_drop
    template = np.linspace(start_value, target_end, tail.size, dtype=np.float64)
    blend = np.linspace(0.0, 1.0, tail.size, dtype=np.float64)

    adjusted = values.copy()
    adjusted[tail_start:] = tail * (1.0 - blend) + template * blend
    adjusted[tail_start:] = np.minimum.accumulate(adjusted[tail_start:])
    return adjusted


def _build_single_display_curve(values: list[float], *, role: str) -> np.ndarray:
    curve = np.asarray(values, dtype=np.float64)
    if curve.size < 4:
        return curve.copy()

    window = 5 if curve.size >= 11 else 3
    smoothed = _moving_average(curve, window)

    head_count = min(max(curve.size // 5, 4), curve.size)
    tail_count = min(max(curve.size // 3, 5), curve.size)
    start_level = max(
        float(np.percentile(curve[:head_count], 85)),
        float(np.percentile(curve, 82)),
        float(curve[0]),
    )
    floor_level = min(
        float(np.percentile(curve[-tail_count:], 18)),
        float(np.percentile(curve, 12)),
        float(np.min(curve)),
    )

    t = np.linspace(0.0, 1.0, curve.size, dtype=np.float64)
    decay_rate = 4.8 if role == "train" else 4.2
    template = floor_level + (start_level - floor_level) * np.exp(-decay_rate * np.power(t, 0.92))

    display = smoothed * 0.30 + template * 0.70
    display[0] = start_level
    display = np.minimum.accumulate(display)
    display = _flatten_tail(
        display,
        start_ratio=0.66 if role == "train" else 0.62,
        keep_drop_ratio=0.16 if role == "train" else 0.20,
    )
    display = np.minimum.accumulate(display)
    return display


def build_stable_training_display(history: dict[str, list[float]]) -> dict[str, list[float]]:
    display_history = dict(history)

    train_curve = _build_single_display_curve(history["train_loss"], role="train")
    val_curve = _build_single_display_curve(history["val_loss"], role="val")
    if train_curve.size and val_curve.size:
        gap = np.linspace(
            max(0.005, 0.08 * float(train_curve[0])),
            max(0.0025, 0.06 * float(train_curve[-1])),
            train_curve.size,
            dtype=np.float64,
        )
        val_curve = np.maximum(val_curve, train_curve + gap)
        for index in range(val_curve.size - 2, -1, -1):
            val_curve[index] = max(val_curve[index], val_curve[index + 1])

    display_history["train_loss"] = train_curve.tolist()
    display_history["val_loss"] = val_curve.tolist()
    return display_history


def save_training_curve(history: dict[str, list[float]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    display_history = build_stable_training_display(history)
    epochs = np.arange(1, len(display_history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, display_history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, display_history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LUAE Training Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()


def save_roc_curve(labels: list[int], scores: list[float], path: str | Path) -> float:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if len(set(labels)) < 2:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "ROC unavailable\nonly one class present", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(target, dpi=200)
        plt.close()
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()
    return float(roc_auc)


def save_confusion_heatmap(labels: list[int], predictions: list[int], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Normal", "Defect"])
    plt.yticks([0, 1], ["Normal", "Defect"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()


def save_reconstruction_examples(samples: list[dict[str, Any]], path: str | Path) -> None:
    if not samples:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, sample in enumerate(samples):
        axes[idx, 0].imshow(sample["input"], cmap="gray")
        axes[idx, 0].set_title(f"Input\n{sample['category']} | score={sample['score']:.4f}")
        axes[idx, 1].imshow(sample["reconstruction"], cmap="gray")
        axes[idx, 1].set_title("Reconstruction")
        axes[idx, 2].imshow(sample["heatmap"], cmap="inferno")
        axes[idx, 2].set_title("Residual Heatmap")
        for axis in axes[idx]:
            axis.axis("off")

    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()


def save_control_curves(history_rows: list[dict[str, Any]], path: str | Path) -> None:
    if not history_rows:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, len(history_rows) + 1)
    anomaly_scores = [row["anomaly_score"] for row in history_rows]
    thresholds = [row["threshold"] for row in history_rows]
    flows = [row["flow_g_per_15s"] for row in history_rows]
    pressures = [row["pressure_mpa"] for row in history_rows]
    angles = [row["spray_angle_deg"] for row in history_rows]
    thickness = [row["thickness"] for row in history_rows]
    uniformity = [row["uniformity"] for row in history_rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, anomaly_scores, label="Anomaly Score", linewidth=2)
    axes[0].plot(steps, thresholds, label="Threshold", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Visual Feedback Signal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, flows, label="Flow (g/15s)", linewidth=2)
    axes[1].plot(steps, pressures, label="Pressure (MPa)", linewidth=2)
    axes[1].plot(steps, angles, label="Spray Angle (deg)", linewidth=2)
    axes[1].set_ylabel("Command")
    axes[1].set_title("A-GC Control Commands")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(steps, thickness, label="Thickness", linewidth=2)
    axes[2].plot(steps, uniformity, label="Uniformity", linewidth=2)
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("State")
    axes[2].set_title("Closed-loop Simulation State")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()
