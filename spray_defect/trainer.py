from __future__ import annotations

import copy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import amp
from torch.optim import Adam
from tqdm import tqdm

from .anomaly import assign_severity, compute_batch_scores, estimate_threshold, extract_region_features
from .config import build_scheduler, choose_device, configure_reproducibility, ensure_dir, save_csv, save_json
from .data import build_dataloaders
from .losses import MSESSIMLoss
from .models import LightweightUNetAutoEncoder, count_parameters
from .visualization import (
    save_confusion_heatmap,
    save_reconstruction_examples,
    save_roc_curve,
    save_training_curve,
)


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MSESSIMLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = amp.GradScaler(device.type, enabled=amp_enabled and is_train)

    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_items = 0

    progress = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=amp_enabled):
            reconstructions = model(images)
            loss, parts = criterion(reconstructions, images)

        if is_train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_mse += parts["mse"] * batch_size
        total_ssim += parts["ssim"] * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.detach().cpu().item():.4f}")

    return total_loss / total_items, total_mse / total_items, total_ssim / total_items


def _collect_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    scoring_config: dict[str, Any],
    device: torch.device,
    threshold: float | None = None,
    collect_examples: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    total_latency = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", leave=False):
            images = batch["image"].to(device)

            start = perf_counter()
            reconstructions = model(images)
            total_latency += perf_counter() - start

            (
                scores,
                residual_mean,
                residual_std,
                ssim_score,
                psnr_score,
                anomaly_maps,
                peak_scores,
            ) = compute_batch_scores(images, reconstructions, scoring_config)

            image_array = images.detach().cpu().numpy()
            recon_array = reconstructions.detach().cpu().numpy()

            labels = batch["label"].tolist()
            paths = batch["path"]
            categories = batch["category"]
            sample_ids = batch["sample_id"]

            for index in range(len(labels)):
                row = {
                    "sample_id": sample_ids[index],
                    "path": paths[index],
                    "label": int(labels[index]),
                    "category": categories[index],
                    "anomaly_score": float(scores[index]),
                    "residual_mean": float(residual_mean[index]),
                    "residual_std": float(residual_std[index]),
                    "ssim_score": float(ssim_score[index]),
                    "psnr_score": float(psnr_score[index]),
                    "peak_score": float(peak_scores[index]),
                }

                if threshold is not None:
                    features = extract_region_features(anomaly_maps[index], scoring_config)
                    predicted_label = int(scores[index] > threshold)
                    severity = assign_severity(float(scores[index]), threshold, float(features["defect_ratio"]))
                    row.update(
                        {
                            "threshold": float(threshold),
                            "predicted_label": predicted_label,
                            "defect_ratio": float(features["defect_ratio"]),
                            "severity": severity,
                            "centroid_x": float(features["centroid_x"]),
                            "centroid_y": float(features["centroid_y"]),
                            "bbox_x": int(features["bbox_x"]),
                            "bbox_y": int(features["bbox_y"]),
                            "bbox_w": int(features["bbox_w"]),
                            "bbox_h": int(features["bbox_h"]),
                        }
                    )

                rows.append(row)

                if len(examples) < collect_examples:
                    examples.append(
                        {
                            "input": image_array[index, 0],
                            "reconstruction": recon_array[index, 0],
                            "heatmap": anomaly_maps[index],
                            "score": float(scores[index]),
                            "category": categories[index],
                        }
                    )

    avg_latency_ms = 1000.0 * total_latency / max(len(loader.dataset), 1)
    return rows, examples, avg_latency_ms


def train_and_evaluate(config: dict[str, Any], max_test_samples: int | None = None) -> dict[str, Any]:
    reproducibility_state = configure_reproducibility(config["seed"], config.get("reproducibility"))
    loaders = build_dataloaders(config, max_test_samples=max_test_samples)

    output_root = ensure_dir(config["paths"]["output_root"])
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    figures_dir = ensure_dir(output_root / "figures")
    metrics_dir = ensure_dir(output_root / "metrics")

    training_config = config["training"]
    device = choose_device(training_config.get("device", "auto"))
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = (
            bool(training_config.get("cudnn_benchmark", False)) and not reproducibility_state["deterministic"]
        )
    model_config = config.get("model", {})
    base_channels = int(model_config.get("base_channels", 32))
    model = LightweightUNetAutoEncoder(base_channels=base_channels).to(device)
    criterion = MSESSIMLoss(
        mse_weight=config["loss"]["mse_weight"],
        ssim_weight=config["loss"]["ssim_weight"],
    )
    optimizer = Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, training_config, epochs=training_config["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [], "train_ssim": [], "val_ssim": []}
    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = int(training_config.get("early_stopping_patience", 15))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_mse, train_ssim = _run_epoch(
            model, loaders["train"], criterion, optimizer, device, amp_enabled
        )
        val_loss, val_mse, val_ssim = _run_epoch(model, loaders["val"], criterion, None, device, amp_enabled)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_ssim"].append(train_ssim)
        history["val_ssim"].append(val_ssim)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_ssim={train_ssim:.4f} | val_ssim={val_ssim:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"早停触发：连续 {patience} 个 epoch 未提升。")
            break

    if best_state is None:
        raise RuntimeError("训练失败，未得到可用模型。")

    model.load_state_dict(best_state)

    val_rows, _, _ = _collect_scores(model, loaders["val"], config["scoring"], device)
    val_scores = np.array([row["anomaly_score"] for row in val_rows], dtype=np.float32)
    threshold = estimate_threshold(val_scores, config["scoring"])

    test_rows, examples, avg_latency_ms = _collect_scores(
        model,
        loaders["test"],
        config["scoring"],
        device,
        threshold=threshold,
        collect_examples=config["visualization"].get("num_examples", 6),
    )

    labels = [int(row["label"]) for row in test_rows]
    predictions = [int(row["predicted_label"]) for row in test_rows]
    scores = [float(row["anomaly_score"]) for row in test_rows]

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold": float(threshold),
        "avg_inference_latency_ms": float(avg_latency_ms),
        "parameter_count": int(count_parameters(model)),
        "device": str(device),
        "base_channels": base_channels,
        "scheduler_type": str(training_config.get("scheduler", {}).get("type", "cosine")),
        "seed": int(config["seed"]),
        "deterministic": bool(reproducibility_state["deterministic"]),
    }

    checkpoint_path = checkpoints_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "threshold": threshold,
            "config": config,
            "metrics": metrics,
        },
        checkpoint_path,
    )

    save_training_curve(history, figures_dir / "training_curve.png")
    save_roc_curve(labels, scores, figures_dir / "roc_curve.png")
    save_confusion_heatmap(labels, predictions, figures_dir / "confusion_matrix.png")
    save_reconstruction_examples(examples, figures_dir / "reconstruction_examples.png")

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_csv(test_rows, metrics_dir / "test_predictions.csv")
    save_json({"history": history}, metrics_dir / "training_history.json")

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "output_root": str(output_root),
    }
