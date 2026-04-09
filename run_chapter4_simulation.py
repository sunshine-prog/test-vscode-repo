from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from spray_defect.anomaly import (
    DetectionSignal,
    aggregate_patch_rows,
    assign_severity,
    compute_batch_scores,
    estimate_threshold,
    extract_region_features,
)
from spray_defect.config import choose_device, ensure_dir, load_yaml, save_csv, save_json, set_seed
from spray_defect.control import AdaptiveGainController, SprayProcessSimulator
from spray_defect.data_v2 import build_dataloaders_v2
from spray_defect.models import LightweightUNetAutoEncoder
from spray_defect.visualization import save_control_curves
from spray_defect.chapter4_pipeline import main as benchmark_main


def _load_model(chapter3_config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> tuple[LightweightUNetAutoEncoder, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LightweightUNetAutoEncoder().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.0))

    if threshold <= 0.0:
        loaders = build_dataloaders_v2(chapter3_config)
        scores: list[float] = []
        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                reconstructions = model(images)
                batch_scores, _, _, _, _, _, _ = compute_batch_scores(images, reconstructions, chapter3_config["scoring"])
                scores.extend(batch_scores.tolist())
        threshold = estimate_threshold(torch.tensor(scores, dtype=torch.float32).numpy(), chapter3_config["scoring"])

    return model, threshold


def _build_signals(
    model: LightweightUNetAutoEncoder,
    loader: torch.utils.data.DataLoader,
    scoring_config: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> tuple[list[DetectionSignal], float]:
    patch_rows: list[dict[str, Any]] = []
    total_latency = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            start = time.perf_counter()
            reconstructions = model(images)
            total_latency += time.perf_counter() - start

            scores, residual_mean, residual_std, ssim_score, psnr_score, anomaly_maps, peak_scores = compute_batch_scores(
                images, reconstructions, scoring_config
            )

            labels = batch["label"].tolist()
            categories = batch["category"]
            paths = batch["path"]
            sample_ids = batch["sample_id"]

            for index in range(len(labels)):
                region = extract_region_features(anomaly_maps[index], scoring_config)
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
                        "defect_ratio": float(region["defect_ratio"]),
                        "centroid_x": float(region["centroid_x"]),
                        "centroid_y": float(region["centroid_y"]),
                        "bbox_x": int(region["bbox_x"]),
                        "bbox_y": int(region["bbox_y"]),
                        "bbox_w": int(region["bbox_w"]),
                        "bbox_h": int(region["bbox_h"]),
                    }
                )

    rows = aggregate_patch_rows(patch_rows, scoring_config, image_size=getattr(loader.dataset, "image_size", 256))
    signals: list[DetectionSignal] = []
    for row in rows:
        predicted_label = int(float(row["anomaly_score"]) > threshold)
        severity = assign_severity(float(row["anomaly_score"]), threshold, float(row["defect_ratio"]))
        signals.append(
            DetectionSignal(
                sample_id=str(row["sample_id"]),
                path=str(row["path"]),
                label=int(row["label"]),
                category=str(row["category"]),
                anomaly_score=float(row["anomaly_score"]),
                threshold=float(threshold),
                predicted_label=predicted_label,
                residual_mean=float(row["residual_mean"]),
                residual_std=float(row["residual_std"]),
                ssim_score=float(row["ssim_score"]),
                psnr_score=float(row["psnr_score"]),
                defect_ratio=float(row["defect_ratio"]),
                severity=severity,
                centroid_x=float(row["centroid_x"]),
                centroid_y=float(row["centroid_y"]),
                bbox_x=int(row["bbox_x"]),
                bbox_y=int(row["bbox_y"]),
                bbox_w=int(row["bbox_w"]),
                bbox_h=int(row["bbox_h"]),
            )
        )

    avg_latency_ms = 1000.0 * total_latency / max(len(signals), 1)
    return signals, avg_latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="第四章：A-GC 闭环控制仿真")
    parser.add_argument("--config", default="configs/chapter4_agc.yaml", help="第四章配置文件路径")
    parser.add_argument("--device", default=None, help="覆盖推理设备")
    parser.add_argument("--max-samples", type=int, default=None, help="只取部分测试样本跑闭环仿真")
    args = parser.parse_args()

    chapter4_config = copy.deepcopy(load_yaml(args.config))
    chapter3_config = load_yaml(chapter4_config["paths"]["chapter3_config"])
    set_seed(chapter4_config["seed"])

    if args.device is not None:
        chapter3_config["training"]["device"] = args.device

    device = choose_device(chapter3_config["training"].get("device", "auto"))
    model, threshold = _load_model(chapter3_config, chapter4_config["paths"]["chapter3_checkpoint"], device)

    max_samples = args.max_samples if args.max_samples is not None else chapter4_config["simulation"].get("max_samples")
    loaders = build_dataloaders_v2(chapter3_config, max_test_samples=max_samples)
    signals, avg_latency_ms = _build_signals(model, loaders["test"], chapter3_config["scoring"], device, threshold)

    output_root = ensure_dir(chapter4_config["paths"]["output_root"])
    figures_dir = ensure_dir(output_root / "figures")

    controller = AdaptiveGainController(chapter4_config["controller"])
    simulator = SprayProcessSimulator(
        chapter4_config["plant"],
        chapter4_config["controller"]["base_parameters"],
        seed=chapter4_config["seed"],
    )

    rows: list[dict[str, Any]] = []
    for cycle, signal in enumerate(signals, start=1):
        command = controller.step(signal)
        state = simulator.step(command, signal)
        rows.append(
            {
                "cycle": cycle,
                "sample_id": signal.sample_id,
                "path": signal.path,
                "true_label": signal.label,
                "predicted_label": signal.predicted_label,
                "severity": signal.severity,
                "anomaly_score": signal.anomaly_score,
                "threshold": signal.threshold,
                "residual_mean": signal.residual_mean,
                "defect_ratio": signal.defect_ratio,
                "flow_g_per_15s": command.flow_g_per_15s,
                "pressure_mpa": command.pressure_mpa,
                "spray_angle_deg": command.spray_angle_deg,
                "action": command.action,
                "filtered_error": command.filtered_error,
                "delta_error": command.delta_error,
                "thickness": state["thickness"],
                "steady_error": state["steady_error"],
                "uniformity": state["uniformity"],
                "centroid_x": signal.centroid_x,
                "centroid_y": signal.centroid_y,
            }
        )

    summary = {
        "num_cycles": len(rows),
        "avg_inference_latency_ms": avg_latency_ms,
        "avg_flow_g_per_15s": float(sum(row["flow_g_per_15s"] for row in rows) / max(len(rows), 1)),
        "avg_pressure_mpa": float(sum(row["pressure_mpa"] for row in rows) / max(len(rows), 1)),
        "avg_spray_angle_deg": float(sum(row["spray_angle_deg"] for row in rows) / max(len(rows), 1)),
        "avg_uniformity": float(sum(row["uniformity"] for row in rows) / max(len(rows), 1)),
        "avg_abs_steady_error": float(sum(abs(row["steady_error"]) for row in rows) / max(len(rows), 1)),
        "alarm_count": int(sum(1 for row in rows if row["action"] == "alarm_and_global_respray")),
        "local_respray_count": int(sum(1 for row in rows if row["action"] == "local_respray")),
        "hold_count": int(sum(1 for row in rows if row["action"] == "hold")),
    }

    save_csv(rows, output_root / "control_log.csv")
    save_json(summary, output_root / "control_summary.json")
    save_control_curves(rows, figures_dir / "control_curves.png")

    print("\n第四章闭环仿真完成。")
    print(f"控制日志: {output_root / 'control_log.csv'}")
    print(f"统计摘要: {output_root / 'control_summary.json'}")
    print(f"平均单样本推理时延: {avg_latency_ms:.2f} ms")
    print(
        "控制摘要: "
        f"avg_uniformity={summary['avg_uniformity']:.4f}, "
        f"avg_abs_steady_error={summary['avg_abs_steady_error']:.4f}, "
        f"alarm_count={summary['alarm_count']}"
    )


if __name__ == "__main__":
    benchmark_main()
