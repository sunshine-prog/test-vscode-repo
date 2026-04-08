from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def save_training_curve(history: dict[str, list[float]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
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
