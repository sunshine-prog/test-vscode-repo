from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import amp
from torchvision.models import ResNet18_Weights, resnet18, wide_resnet50_2

from spray_defect.anomaly import estimate_threshold_with_labels, extract_region_features
from spray_defect.config import choose_device, ensure_dir, load_yaml, save_csv, save_json
from spray_defect.data_v2 import build_dataloaders_v2
from spray_defect.visualization import save_confusion_heatmap, save_roc_curve


class ResNet18FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        normalized = backbone_name.strip().lower()
        if normalized == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif normalized == "wide_resnet50_2":
            backbone = wide_resnet50_2(weights=None)
            checkpoint_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing local wide_resnet50_2 weights: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            backbone.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name!r}")
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb = x.repeat(1, 3, 1, 1)
        rgb = (rgb - self.mean) / self.std
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        embedding = torch.flatten(F.adaptive_avg_pool2d(layer4, output_size=1), start_dim=1)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "embedding": embedding,
        }


def _compute_classification_metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, float]:
    auc_value = float("nan") if len(set(labels)) < 2 else float(roc_auc_score(labels, scores))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc_value,
    }


def _full_image_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("patching", {})["enabled"] = False
    updated.setdefault("patching", {})["cache_enabled"] = False
    updated.setdefault("patching", {})["cache_dir"] = None
    return updated


def _aggregate_scalar_rows(rows: list[dict[str, Any]], mode: str, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["path"]), int(row["label"]))].append(row)

    aggregated: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        scores = sorted([float(row["patch_score"]) for row in group_rows], reverse=True)
        if mode == "max":
            score = float(scores[0])
        elif mode == "mean":
            score = float(np.mean(scores))
        else:
            score = float(np.mean(scores[: min(max(top_k, 1), len(scores))]))

        top_row = max(group_rows, key=lambda row: float(row["patch_score"]))
        aggregated.append(
            {
                "sample_id": str(top_row["sample_id"]),
                "path": str(top_row["path"]),
                "label": int(top_row["label"]),
                "category": str(top_row["category"]),
                "anomaly_score": score,
                "patch_count": len(group_rows),
            }
        )

    aggregated.sort(key=lambda row: str(row["path"]))
    return aggregated


def _aggregate_padim_rows(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    region_defect_weight: float,
    region_bbox_width_weight: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["path"]), int(row["label"]))].append(row)

    aggregated: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        sorted_rows = sorted(group_rows, key=lambda row: float(row["patch_score"]), reverse=True)
        top_rows = sorted_rows[: min(max(top_k, 1), len(sorted_rows))]
        base_score = float(np.mean([float(row["patch_score"]) for row in top_rows]))
        defect_ratio = float(np.mean([float(row["defect_ratio"]) for row in top_rows]))
        bbox_width_ratio = float(np.mean([float(row["bbox_width_ratio"]) for row in top_rows]))
        region_score = region_defect_weight * defect_ratio + region_bbox_width_weight * bbox_width_ratio
        top_row = sorted_rows[0]

        aggregated.append(
            {
                "sample_id": str(top_row["sample_id"]),
                "path": str(top_row["path"]),
                "label": int(top_row["label"]),
                "category": str(top_row["category"]),
                "anomaly_score": base_score + region_score,
                "base_anomaly_score": base_score,
                "defect_ratio": defect_ratio,
                "bbox_width_ratio": bbox_width_ratio,
                "region_score": region_score,
                "patch_count": len(group_rows),
            }
        )

    aggregated.sort(key=lambda row: str(row["path"]))
    return aggregated


def _pick_best_scalar_aggregator(val_rows: list[dict[str, Any]], defect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(val_rows) + list(defect_rows)
    best: dict[str, Any] | None = None
    for mode in ("max", "mean", "topk_mean"):
        for top_k in (1, 3, 5):
            aggregated = _aggregate_scalar_rows(all_rows, mode=mode, top_k=top_k)
            labels = [int(row["label"]) for row in aggregated]
            scores = [float(row["anomaly_score"]) for row in aggregated]
            auc = float(roc_auc_score(labels, scores))
            candidate = {"mode": mode, "top_k": top_k, "val_auc": auc}
            if best is None or auc > float(best["val_auc"]):
                best = candidate
    assert best is not None
    return best


def _pick_best_padim_aggregator(val_rows: list[dict[str, Any]], defect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(val_rows) + list(defect_rows)
    best: dict[str, Any] | None = None
    for top_k in (1, 3, 5):
        for defect_weight in (0.0, 1.5, 3.0, 5.0, 6.75):
            for bbox_width_weight in (0.0, 1.5, 3.0, 5.0):
                aggregated = _aggregate_padim_rows(
                    all_rows,
                    top_k=top_k,
                    region_defect_weight=defect_weight,
                    region_bbox_width_weight=bbox_width_weight,
                )
                labels = [int(row["label"]) for row in aggregated]
                scores = [float(row["anomaly_score"]) for row in aggregated]
                auc = float(roc_auc_score(labels, scores))
                candidate = {
                    "top_k": top_k,
                    "region_defect_weight": defect_weight,
                    "region_bbox_width_weight": bbox_width_weight,
                    "val_auc": auc,
                }
                if best is None or auc > float(best["val_auc"]):
                    best = candidate
    assert best is not None
    return best


def _finalize_predictions(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["threshold"] = float(threshold)
        updated["predicted_label"] = int(float(row["anomaly_score"]) > threshold)
        finalized.append(updated)
    return finalized


def _save_benchmark_outputs(
    output_root: Path,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    figures_dir = ensure_dir(output_root / "figures")
    metrics_dir = ensure_dir(output_root / "metrics")
    labels = [int(row["label"]) for row in predictions]
    prediction_labels = [int(row["predicted_label"]) for row in predictions]
    scores = [float(row["anomaly_score"]) for row in predictions]

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_csv(predictions, metrics_dir / "test_predictions.csv")
    save_roc_curve(labels, scores, figures_dir / "roc_curve.png")
    save_confusion_heatmap(labels, prediction_labels, figures_dir / "confusion_matrix.png")

    return {
        "output_root": str(output_root),
        "metrics": metrics,
    }


def _collect_train_embeddings(
    extractor: ResNet18FeatureExtractor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            features = extractor(images)
            embeddings.append(features["embedding"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


def _collect_resnet18_patch_rows(
    extractor: ResNet18FeatureExtractor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mean_vector: np.ndarray,
    precision_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            embedding = extractor(images)["embedding"].detach().cpu().numpy().astype(np.float32)
            diff = embedding - mean_vector[None, :]
            scores = np.einsum("bd,dk,bk->b", diff, precision_matrix, diff).astype(np.float32)
            for index in range(len(batch["label"])):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "path": batch["path"][index],
                        "label": int(batch["label"][index]),
                        "category": batch["category"][index],
                        "patch_score": float(scores[index]),
                    }
                )
    return rows


def _fit_padim_statistics(
    extractor: ResNet18FeatureExtractor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    embedding_dim: int,
    seed: int,
) -> dict[str, Any]:
    feature_batches: list[np.ndarray] = []
    selected_dims: np.ndarray | None = None
    map_height = 0
    map_width = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            features = extractor(images)
            layer1 = F.adaptive_avg_pool2d(features["layer1"], output_size=features["layer3"].shape[-2:])
            layer2 = F.adaptive_avg_pool2d(features["layer2"], output_size=features["layer3"].shape[-2:])
            layer3 = features["layer3"]
            embedding = torch.cat([layer1, layer2, layer3], dim=1).detach().cpu().numpy().astype(np.float32)
            if selected_dims is None:
                rng = np.random.default_rng(seed)
                channel_count = embedding.shape[1]
                selected_dims = np.sort(rng.choice(channel_count, size=min(embedding_dim, channel_count), replace=False))
                map_height, map_width = embedding.shape[2], embedding.shape[3]
            feature_batches.append(embedding[:, selected_dims, :, :])

    train_features = np.concatenate(feature_batches, axis=0)
    sample_count, channel_count, _, _ = train_features.shape
    flattened = np.transpose(train_features, (0, 2, 3, 1)).reshape(sample_count, map_height * map_width, channel_count)
    mean = flattened.mean(axis=0).astype(np.float32)
    inv_covariances = np.empty((map_height * map_width, channel_count, channel_count), dtype=np.float32)
    regularizer = np.eye(channel_count, dtype=np.float32) * 0.01

    for location in range(map_height * map_width):
        vectors = flattened[:, location, :]
        covariance = np.cov(vectors, rowvar=False).astype(np.float32)
        inv_covariances[location] = np.linalg.pinv(covariance + regularizer).astype(np.float32)

    return {
        "selected_dims": selected_dims,
        "mean": mean,
        "inv_covariances": inv_covariances,
        "map_height": map_height,
        "map_width": map_width,
    }


def _collect_padim_patch_rows(
    extractor: ResNet18FeatureExtractor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    statistics: dict[str, Any],
    score_reduction: str = "mean",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_dims = statistics["selected_dims"]
    mean = statistics["mean"]
    inv_covariances = statistics["inv_covariances"]
    map_height = int(statistics["map_height"])
    map_width = int(statistics["map_width"])

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            features = extractor(images)
            layer1 = F.adaptive_avg_pool2d(features["layer1"], output_size=features["layer3"].shape[-2:])
            layer2 = F.adaptive_avg_pool2d(features["layer2"], output_size=features["layer3"].shape[-2:])
            layer3 = features["layer3"]
            embedding = torch.cat([layer1, layer2, layer3], dim=1)[:, selected_dims, :, :]
            embedding_np = np.transpose(
                embedding.detach().cpu().numpy().astype(np.float32),
                (0, 2, 3, 1),
            ).reshape(embedding.size(0), map_height * map_width, len(selected_dims))
            diff = embedding_np - mean[None, :, :]
            distances = np.einsum("bld,ldk,blk->bl", diff, inv_covariances, diff).astype(np.float32)
            score_maps = distances.reshape(embedding.size(0), map_height, map_width)

            for index in range(len(batch["label"])):
                anomaly_map = cv2.resize(score_maps[index], (images.size(-1), images.size(-2)), interpolation=cv2.INTER_CUBIC)
                anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=4.0, sigmaY=4.0)
                region = extract_region_features(anomaly_map, {"map_std_factor": 2.0, "min_region_area": 20})
                patch_w = int(batch["patch_w"][index])
                base_width = max(int(batch["base_width"][index]), 1)
                mapped_bbox_w = float(int(region["bbox_w"]) * patch_w / max(images.size(-1), 1))
                if score_reduction == "max":
                    patch_score = float(anomaly_map.max())
                elif score_reduction == "q99":
                    patch_score = float(np.quantile(anomaly_map, 0.99))
                else:
                    patch_score = float(anomaly_map.mean())
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "path": batch["path"][index],
                        "label": int(batch["label"][index]),
                        "category": batch["category"][index],
                        "patch_score": patch_score,
                        "defect_ratio": float(region["defect_ratio"]),
                        "bbox_width_ratio": mapped_bbox_w / base_width,
                    }
                )
    return rows


def run_resnet18_baseline(config: dict[str, Any], output_root: Path, max_test_samples: int | None = None) -> dict[str, Any]:
    benchmark_config = _full_image_config(config)
    loaders = build_dataloaders_v2(benchmark_config, max_test_samples=max_test_samples)
    device = choose_device(config["training"].get("device", "auto"))
    extractor = ResNet18FeatureExtractor().to(device).eval()

    train_embeddings = _collect_train_embeddings(extractor, loaders["train"], device)
    covariance = LedoitWolf().fit(train_embeddings)
    mean_vector = covariance.location_.astype(np.float32)
    precision_matrix = covariance.precision_.astype(np.float32)

    val_rows = _collect_resnet18_patch_rows(extractor, loaders["val"], device, mean_vector, precision_matrix)
    val_defect_rows = _collect_resnet18_patch_rows(extractor, loaders["val_defect"], device, mean_vector, precision_matrix)
    aggregator = _pick_best_scalar_aggregator(val_rows, val_defect_rows)

    val_image_rows = _aggregate_scalar_rows(val_rows + val_defect_rows, mode=aggregator["mode"], top_k=int(aggregator["top_k"]))
    threshold_scores = np.array([float(row["anomaly_score"]) for row in val_image_rows], dtype=np.float32)
    threshold_labels = np.array([int(row["label"]) for row in val_image_rows], dtype=np.int64)
    threshold = estimate_threshold_with_labels(
        threshold_scores,
        threshold_labels,
        {
            "threshold_method": "f1_search",
            "threshold_search_min_percentile": 80.0,
            "threshold_search_max_percentile": 97.5,
            "threshold_search_num_steps": 71,
        },
    )

    test_rows = _collect_resnet18_patch_rows(extractor, loaders["test"], device, mean_vector, precision_matrix)
    test_image_rows = _aggregate_scalar_rows(test_rows, mode=aggregator["mode"], top_k=int(aggregator["top_k"]))
    predictions = _finalize_predictions(test_image_rows, threshold=threshold)

    labels = [int(row["label"]) for row in predictions]
    prediction_labels = [int(row["predicted_label"]) for row in predictions]
    scores = [float(row["anomaly_score"]) for row in predictions]
    metrics = _compute_classification_metrics(labels, prediction_labels, scores)
    metrics.update(
        {
            "threshold": float(threshold),
            "device": str(device),
            "method": "ResNet18",
            "backbone": "resnet18_imagenet",
            "aggregation_mode": str(aggregator["mode"]),
            "aggregation_top_k": int(aggregator["top_k"]),
            "val_auc": float(aggregator["val_auc"]),
        }
    )

    return _save_benchmark_outputs(output_root, predictions, metrics)


def run_padim(config: dict[str, Any], output_root: Path, max_test_samples: int | None = None) -> dict[str, Any]:
    benchmark_config = _full_image_config(config)
    loaders = build_dataloaders_v2(benchmark_config, max_test_samples=max_test_samples)
    device = choose_device(config["training"].get("device", "auto"))
    extractor = ResNet18FeatureExtractor(backbone_name="wide_resnet50_2").to(device).eval()

    statistics = _fit_padim_statistics(
        extractor,
        loaders["train"],
        device,
        embedding_dim=100,
        seed=int(config.get("seed", 42)),
    )

    val_rows = _collect_padim_patch_rows(extractor, loaders["val"], device, statistics, score_reduction="mean")
    val_defect_rows = _collect_padim_patch_rows(extractor, loaders["val_defect"], device, statistics, score_reduction="mean")
    aggregator = _pick_best_padim_aggregator(val_rows, val_defect_rows)
    val_image_rows = _aggregate_padim_rows(
        val_rows + val_defect_rows,
        top_k=int(aggregator["top_k"]),
        region_defect_weight=float(aggregator["region_defect_weight"]),
        region_bbox_width_weight=float(aggregator["region_bbox_width_weight"]),
    )
    threshold_scores = np.array([float(row["anomaly_score"]) for row in val_image_rows], dtype=np.float32)
    threshold_labels = np.array([int(row["label"]) for row in val_image_rows], dtype=np.int64)
    threshold = estimate_threshold_with_labels(
        threshold_scores,
        threshold_labels,
        {
            "threshold_method": "f1_search",
            "threshold_search_min_percentile": 80.0,
            "threshold_search_max_percentile": 97.5,
            "threshold_search_num_steps": 71,
        },
    )

    test_rows = _collect_padim_patch_rows(extractor, loaders["test"], device, statistics, score_reduction="mean")
    test_image_rows = _aggregate_padim_rows(
        test_rows,
        top_k=int(aggregator["top_k"]),
        region_defect_weight=float(aggregator["region_defect_weight"]),
        region_bbox_width_weight=float(aggregator["region_bbox_width_weight"]),
    )
    predictions = _finalize_predictions(test_image_rows, threshold=threshold)

    labels = [int(row["label"]) for row in predictions]
    prediction_labels = [int(row["predicted_label"]) for row in predictions]
    scores = [float(row["anomaly_score"]) for row in predictions]
    metrics = _compute_classification_metrics(labels, prediction_labels, scores)
    metrics.update(
        {
            "threshold": float(threshold),
            "device": str(device),
            "method": "PaDiM",
            "backbone": "wide_resnet50_2_imagenet",
            "embedding_dim": int(len(statistics["selected_dims"])),
            "aggregation_top_k": int(aggregator["top_k"]),
            "region_defect_weight": float(aggregator["region_defect_weight"]),
            "region_bbox_width_weight": float(aggregator["region_bbox_width_weight"]),
            "score_reduction": "mean",
            "val_auc": float(aggregator["val_auc"]),
        }
    )

    return _save_benchmark_outputs(output_root, predictions, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter 3 feature baselines: ResNet18 and PaDiM")
    parser.add_argument("--config", default="configs/chapter3_luae.yaml", help="Path to chapter 3 config")
    parser.add_argument("--method", choices=("resnet18", "padim", "both"), default="both", help="Benchmark method to run")
    parser.add_argument("--output-root", default="D:/pythonProject2/outputs/chapter3_feature_benchmarks", help="Output root")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for quick checks")
    args = parser.parse_args()

    config = load_yaml(args.config)
    base_output_root = ensure_dir(Path(args.output_root))

    if args.method in {"resnet18", "both"}:
        resnet_output = ensure_dir(base_output_root / "resnet18")
        result = run_resnet18_baseline(copy.deepcopy(config), resnet_output, max_test_samples=args.max_test_samples)
        metrics = result["metrics"]
        print(
            f"ResNet18 | "
            f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f} | AUC={metrics['auc']:.4f}"
        )

    if args.method in {"padim", "both"}:
        padim_output = ensure_dir(base_output_root / "padim")
        result = run_padim(copy.deepcopy(config), padim_output, max_test_samples=args.max_test_samples)
        metrics = result["metrics"]
        print(
            f"PaDiM | "
            f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f} | AUC={metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
