from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .losses import psnr_from_mse, structural_similarity


@dataclass
class DetectionSignal:
    sample_id: str
    path: str
    label: int
    category: str
    anomaly_score: float
    threshold: float
    predicted_label: int
    residual_mean: float
    residual_std: float
    ssim_score: float
    psnr_score: float
    defect_ratio: float
    severity: str
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


def compute_batch_scores(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    scoring_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = torch.abs(inputs - reconstructions)
    residual_mean = residual.mean(dim=(1, 2, 3))
    residual_std = residual.std(dim=(1, 2, 3))
    mse = F.mse_loss(reconstructions, inputs, reduction="none").mean(dim=(1, 2, 3))
    ssim_score, ssim_map = structural_similarity(reconstructions, inputs, full=True)
    psnr_score = psnr_from_mse(mse)

    anomaly_map = 0.5 * residual.mean(dim=1) + 0.5 * (1.0 - ssim_map.mean(dim=1))
    peak_percentile = float(scoring_config.get("peak_percentile", 99.0)) / 100.0
    peak_score = torch.quantile(anomaly_map.view(anomaly_map.size(0), -1), peak_percentile, dim=1)
    anomaly_score = (
        scoring_config["residual_weight"] * residual_mean
        + scoring_config["ssim_weight"] * (1.0 - ssim_score)
        + scoring_config["psnr_weight"] * (1.0 / (psnr_score + 1.0))
        + scoring_config.get("peak_weight", 0.0) * peak_score
    )

    return (
        anomaly_score.detach().cpu().numpy(),
        residual_mean.detach().cpu().numpy(),
        residual_std.detach().cpu().numpy(),
        ssim_score.detach().cpu().numpy(),
        psnr_score.detach().cpu().numpy(),
        anomaly_map.detach().cpu().numpy(),
        peak_score.detach().cpu().numpy(),
    )


def estimate_threshold(scores: np.ndarray, scoring_config: dict[str, Any]) -> float:
    method = scoring_config.get("threshold_method", "mean_std")
    if method == "percentile":
        return float(np.percentile(scores, scoring_config.get("threshold_percentile", 97.5)))

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean + scoring_config.get("threshold_std_factor", 3.0) * std


def estimate_threshold_with_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    scoring_config: dict[str, Any],
) -> float:
    method = str(scoring_config.get("threshold_method", "mean_std")).strip().lower()
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if method not in {"f1_search", "f1"} or scores.size == 0 or labels.size != scores.size:
        return estimate_threshold(scores, scoring_config)

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        normal_scores = scores[labels == 0]
        return estimate_threshold(normal_scores if normal_scores.size else scores, scoring_config)

    normal_scores = scores[labels == 0]
    candidate_source = normal_scores if normal_scores.size else scores
    min_percentile = float(scoring_config.get("threshold_search_min_percentile", 80.0))
    max_percentile = float(scoring_config.get("threshold_search_max_percentile", 97.5))
    num_steps = int(scoring_config.get("threshold_search_num_steps", 71))
    percentiles = np.linspace(min_percentile, max_percentile, max(num_steps, 2))
    thresholds = np.unique(np.percentile(candidate_source, percentiles))
    if thresholds.size == 0:
        return estimate_threshold(scores, scoring_config)

    best_threshold = float(thresholds[0])
    best_f1 = float("-inf")
    best_recall = float("-inf")
    best_precision = float("-inf")
    for threshold in thresholds:
        predictions = (scores > float(threshold)).astype(np.int64)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        true_positive = int(np.sum((predictions == 1) & (labels == 1)))
        predicted_positive = int(np.sum(predictions == 1))
        actual_positive = int(np.sum(labels == 1))
        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(actual_positive, 1)
        if (
            f1 > best_f1
            or (np.isclose(f1, best_f1) and recall > best_recall)
            or (np.isclose(f1, best_f1) and np.isclose(recall, best_recall) and precision > best_precision)
        ):
            best_threshold = float(threshold)
            best_f1 = f1
            best_recall = recall
            best_precision = precision

    return best_threshold


def extract_region_features(anomaly_map: np.ndarray, scoring_config: dict[str, Any]) -> dict[str, float | int]:
    threshold = float(anomaly_map.mean() + scoring_config.get("map_std_factor", 2.0) * anomaly_map.std())
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if float(anomaly_map.max() - anomaly_map.min()) < 1e-8:
        binary = np.zeros_like(normalized)
    else:
        threshold_norm = int(
            np.clip(
                255.0 * (threshold - float(anomaly_map.min())) / (float(anomaly_map.max() - anomaly_map.min()) + 1e-8),
                0,
                255,
            )
        )
        _, binary = cv2.threshold(normalized, threshold_norm, 255, cv2.THRESH_BINARY)

    min_region_area = int(scoring_config.get("min_region_area", 20))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)

    best_index = -1
    best_area = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_region_area and area > best_area:
            best_area = area
            best_index = index

    if best_index < 0:
        return {
            "defect_ratio": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
        }

    height, width = anomaly_map.shape
    x = int(stats[best_index, cv2.CC_STAT_LEFT])
    y = int(stats[best_index, cv2.CC_STAT_TOP])
    w = int(stats[best_index, cv2.CC_STAT_WIDTH])
    h = int(stats[best_index, cv2.CC_STAT_HEIGHT])
    area = int(stats[best_index, cv2.CC_STAT_AREA])
    centroid_x = float(centroids[best_index][0] / max(width, 1))
    centroid_y = float(centroids[best_index][1] / max(height, 1))
    defect_ratio = float(area / float(height * width))

    return {
        "defect_ratio": defect_ratio,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
    }


def assign_severity(score: float, threshold: float, defect_ratio: float) -> str:
    if score <= threshold or defect_ratio <= 0.0:
        return "normal"

    score_ratio = score / max(threshold, 1e-8)
    if score_ratio < 1.20 and defect_ratio < 0.02:
        return "slight"
    if score_ratio < 1.60 and defect_ratio < 0.08:
        return "medium"
    return "severe"


def _topk_mean(values: list[float], top_k: int) -> float:
    if not values:
        return 0.0
    limit = min(max(top_k, 1), len(values))
    return float(np.mean(sorted(values, reverse=True)[:limit]))


def aggregate_patch_rows(rows: list[dict[str, Any]], scoring_config: dict[str, Any], image_size: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["path"], int(row["label"]))].append(row)

    aggregated_rows: list[dict[str, Any]] = []
    top_k = int(scoring_config.get("aggregate_top_k", 1))
    topk_weight = float(scoring_config.get("aggregate_topk_weight", 0.0))
    max_weight = float(scoring_config.get("aggregate_max_weight", 1.0))
    quantile_weight = float(scoring_config.get("aggregate_quantile_weight", 0.0))
    aggregate_quantile = float(scoring_config.get("aggregate_quantile", 0.9))
    region_defect_weight = float(scoring_config.get("region_defect_weight", 0.0))
    region_bbox_width_weight = float(scoring_config.get("region_bbox_width_weight", 0.0))
    region_bbox_area_weight = float(scoring_config.get("region_bbox_area_weight", 0.0))

    for group_rows in grouped.values():
        scores = np.array([float(row["anomaly_score"]) for row in group_rows], dtype=np.float32)
        sorted_rows = sorted(group_rows, key=lambda row: float(row["anomaly_score"]), reverse=True)
        top_rows = sorted_rows[: min(max(top_k, 1), len(sorted_rows))]
        aggregate_score = topk_weight * _topk_mean([float(row["anomaly_score"]) for row in group_rows], top_k)
        aggregate_score += max_weight * float(scores.max())
        if quantile_weight > 0.0:
            aggregate_score += quantile_weight * float(np.quantile(scores, aggregate_quantile))

        top_row = sorted_rows[0]
        aggregated = dict(top_row)
        aggregated["anomaly_score"] = aggregate_score
        aggregated["patch_count"] = len(group_rows)

        patch_x = int(top_row.get("patch_x", 0))
        patch_y = int(top_row.get("patch_y", 0))
        patch_w = int(top_row.get("patch_w", image_size))
        patch_h = int(top_row.get("patch_h", image_size))
        base_width = max(int(top_row.get("base_width", patch_w)), 1)
        base_height = max(int(top_row.get("base_height", patch_h)), 1)

        aggregated["centroid_x"] = (patch_x + float(top_row["centroid_x"]) * patch_w) / base_width
        aggregated["centroid_y"] = (patch_y + float(top_row["centroid_y"]) * patch_h) / base_height
        aggregated["bbox_x"] = int(patch_x + int(top_row["bbox_x"]) * patch_w / image_size)
        aggregated["bbox_y"] = int(patch_y + int(top_row["bbox_y"]) * patch_h / image_size)
        aggregated["bbox_w"] = int(int(top_row["bbox_w"]) * patch_w / image_size)
        aggregated["bbox_h"] = int(int(top_row["bbox_h"]) * patch_h / image_size)

        bbox_width_ratios: list[float] = []
        bbox_height_ratios: list[float] = []
        bbox_area_ratios: list[float] = []
        defect_ratios: list[float] = []
        for row in top_rows:
            row_patch_w = int(row.get("patch_w", image_size))
            row_patch_h = int(row.get("patch_h", image_size))
            row_base_width = max(int(row.get("base_width", row_patch_w)), 1)
            row_base_height = max(int(row.get("base_height", row_patch_h)), 1)
            mapped_bbox_w = float(int(row["bbox_w"]) * row_patch_w / image_size)
            mapped_bbox_h = float(int(row["bbox_h"]) * row_patch_h / image_size)
            width_ratio = mapped_bbox_w / row_base_width
            height_ratio = mapped_bbox_h / row_base_height
            bbox_width_ratios.append(width_ratio)
            bbox_height_ratios.append(height_ratio)
            bbox_area_ratios.append(width_ratio * height_ratio)
            defect_ratios.append(float(row["defect_ratio"]))

        bbox_width_ratio = float(np.mean(bbox_width_ratios)) if bbox_width_ratios else 0.0
        bbox_height_ratio = float(np.mean(bbox_height_ratios)) if bbox_height_ratios else 0.0
        bbox_area_ratio = bbox_width_ratio * bbox_height_ratio
        region_score = (
            region_defect_weight * (float(np.mean(defect_ratios)) if defect_ratios else 0.0)
            + region_bbox_width_weight * bbox_width_ratio
            + region_bbox_area_weight * (float(np.mean(bbox_area_ratios)) if bbox_area_ratios else 0.0)
        )

        aggregated["base_anomaly_score"] = aggregate_score
        aggregated["bbox_width_ratio"] = bbox_width_ratio
        aggregated["bbox_height_ratio"] = bbox_height_ratio
        aggregated["bbox_area_ratio"] = bbox_area_ratio
        aggregated["region_score"] = region_score
        aggregated["anomaly_score"] = aggregate_score + region_score

        aggregated_rows.append(aggregated)

    aggregated_rows.sort(key=lambda row: str(row["path"]))
    return aggregated_rows
