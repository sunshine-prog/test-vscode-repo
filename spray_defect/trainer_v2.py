from __future__ import annotations

import copy
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import amp
from torch.optim import Adam
from tqdm import tqdm

from .anomaly import (
    aggregate_patch_rows,
    assign_severity,
    compute_batch_scores,
    estimate_threshold,
    extract_region_features,
)
from .config import build_scheduler, choose_device, configure_reproducibility, ensure_dir, save_csv, save_json
from .data_v2 import build_dataloaders_v2
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
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
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

    return total_loss / max(total_items, 1), total_mse / max(total_items, 1), total_ssim / max(total_items, 1)


def _collect_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    scoring_config: dict[str, Any],
    device: torch.device,
    threshold: float | None = None,
    collect_examples: int = 0,
    amp_enabled: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    model.eval()
    patch_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    total_latency = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Infer", leave=False):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")

            start = perf_counter()
            with amp.autocast(device_type=device.type, enabled=amp_enabled):
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
            ) = compute_batch_scores(images.float(), reconstructions.float(), scoring_config)

            need_example_arrays = len(examples) < collect_examples
            image_array = images.detach().cpu().numpy() if need_example_arrays else None
            recon_array = reconstructions.detach().cpu().numpy() if need_example_arrays else None

            labels = batch["label"].tolist()
            paths = batch["path"]
            categories = batch["category"]
            sample_ids = batch["sample_id"]

            for index in range(len(labels)):
                features = extract_region_features(anomaly_maps[index], scoring_config)
                patch_rows.append(
                    {
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
                        "patch_index": int(batch["patch_index"][index]),
                        "patch_x": int(batch["patch_x"][index]),
                        "patch_y": int(batch["patch_y"][index]),
                        "patch_w": int(batch["patch_w"][index]),
                        "patch_h": int(batch["patch_h"][index]),
                        "base_width": int(batch["base_width"][index]),
                        "base_height": int(batch["base_height"][index]),
                        "defect_ratio": float(features["defect_ratio"]),
                        "centroid_x": float(features["centroid_x"]),
                        "centroid_y": float(features["centroid_y"]),
                        "bbox_x": int(features["bbox_x"]),
                        "bbox_y": int(features["bbox_y"]),
                        "bbox_w": int(features["bbox_w"]),
                        "bbox_h": int(features["bbox_h"]),
                        }
                    )

                if need_example_arrays and len(examples) < collect_examples:
                    examples.append(
                        {
                            "input": image_array[index, 0],
                            "reconstruction": recon_array[index, 0],
                            "heatmap": anomaly_maps[index],
                            "score": float(scores[index]),
                            "category": categories[index],
                        }
                    )

    rows = aggregate_patch_rows(patch_rows, scoring_config, image_size=getattr(loader.dataset, "image_size", 256))
    if threshold is not None:
        for row in rows:
            predicted_label = int(float(row["anomaly_score"]) > threshold)
            row["threshold"] = float(threshold)
            row["predicted_label"] = predicted_label
            row["severity"] = assign_severity(float(row["anomaly_score"]), threshold, float(row["defect_ratio"]))

    avg_latency_ms = 1000.0 * total_latency / max(len(rows), 1)
    return rows, examples, avg_latency_ms


def _compute_calibration_auc(
    model: torch.nn.Module,
    normal_loader: torch.utils.data.DataLoader,
    defect_loader: torch.utils.data.DataLoader,
    scoring_config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> float:
    normal_rows, _, _ = _collect_scores(model, normal_loader, scoring_config, device, amp_enabled=amp_enabled)
    defect_rows, _, _ = _collect_scores(model, defect_loader, scoring_config, device, amp_enabled=amp_enabled)
    labels = [0] * len(normal_rows) + [1] * len(defect_rows)
    scores = [float(row["anomaly_score"]) for row in normal_rows] + [float(row["anomaly_score"]) for row in defect_rows]
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def train_and_evaluate_v2(config: dict[str, Any], max_test_samples: int | None = None) -> dict[str, Any]:
    reproducibility_state = configure_reproducibility(config["seed"], config.get("reproducibility"))
    loaders = build_dataloaders_v2(config, max_test_samples=max_test_samples)

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
    norm_type = str(model_config.get("norm_type", "batchnorm"))
    group_count = int(model_config.get("group_count", 8))
    model = LightweightUNetAutoEncoder(
        base_channels=base_channels,
        norm_type=norm_type,
        group_count=group_count,
    ).to(device)
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
    best_monitor_value = float("-inf")
    early_stop_counter = 0
    patience = int(training_config.get("early_stopping_patience", 15))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))
    selection_config = config.get("selection", {})
    monitor = str(selection_config.get("monitor", "val_loss")).strip().lower()
    defect_monitor_loader = loaders.get("val_defect") or loaders.get("calibration_defect")
    defect_monitor_source = "val_defect" if "val_defect" in loaders else ("calibration_defect" if "calibration_defect" in loaders else "none")
    if monitor == "calibration_auc" and defect_monitor_loader is None:
        monitor = "val_loss"
    if monitor == "calibration_auc" and defect_monitor_loader is not None:
        history["calibration_auc"] = []
    elif monitor == "val_loss":
        best_monitor_value = float("inf")

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

        calibration_auc: float | None = None
        if monitor == "calibration_auc" and defect_monitor_loader is not None:
            calibration_auc = _compute_calibration_auc(
                model,
                loaders["val"],
                defect_monitor_loader,
                config["scoring"],
                device,
                amp_enabled,
            )
            history["calibration_auc"].append(calibration_auc)

        message = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_ssim={train_ssim:.4f} | val_ssim={val_ssim:.4f}"
        )
        if calibration_auc is not None:
            message += f" | calib_auc={calibration_auc:.4f}"
        print(message)

        improved = False
        if monitor == "calibration_auc" and calibration_auc is not None and not np.isnan(calibration_auc):
            improved = calibration_auc > best_monitor_value + min_delta
            if improved:
                best_monitor_value = calibration_auc
        else:
            improved = val_loss < best_monitor_value - min_delta
            if improved:
                best_monitor_value = val_loss

        if improved:
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

    val_rows, _, _ = _collect_scores(model, loaders["val"], config["scoring"], device, amp_enabled=amp_enabled)
    val_scores = np.array([row["anomaly_score"] for row in val_rows], dtype=np.float32)
    threshold = estimate_threshold(val_scores, config["scoring"])

    test_rows, examples, avg_latency_ms = _collect_scores(
        model,
        loaders["test"],
        config["scoring"],
        device,
        threshold=threshold,
        collect_examples=config["visualization"].get("num_examples", 6),
        amp_enabled=amp_enabled,
    )

    labels = [int(row["label"]) for row in test_rows]
    predictions = [int(row["predicted_label"]) for row in test_rows]
    scores = [float(row["anomaly_score"]) for row in test_rows]
    auc_value = float("nan") if len(set(labels)) < 2 else float(roc_auc_score(labels, scores))

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc_value,
        "threshold": float(threshold),
        "avg_inference_latency_ms": float(avg_latency_ms),
        "parameter_count": int(count_parameters(model)),
        "device": str(device),
        "base_channels": base_channels,
        "norm_type": norm_type,
        "group_count": group_count,
        "scheduler_type": str(training_config.get("scheduler", {}).get("type", "cosine")),
        "selection_monitor": monitor,
        "defect_monitor_source": defect_monitor_source,
        "best_monitor_value": float(best_monitor_value),
        "seed": int(config["seed"]),
        "deterministic": bool(reproducibility_state["deterministic"]),
        "train_patch_count": len(loaders["train"].dataset),
        "val_patch_count": len(loaders["val"].dataset),
        "test_patch_count": len(loaders["test"].dataset),
    }
    if "calibration_defect" in loaders:
        metrics["calibration_defect_patch_count"] = len(loaders["calibration_defect"].dataset)
    if "val_defect" in loaders:
        metrics["val_defect_patch_count"] = len(loaders["val_defect"].dataset)

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
