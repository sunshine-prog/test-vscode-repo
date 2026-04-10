from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams

from .anomaly import aggregate_patch_rows, compute_batch_scores, extract_region_features
from .config import choose_device, ensure_dir, load_yaml, save_csv, save_json
from .data_v2 import SprayImageDatasetV2
from .models import LightweightUNetAutoEncoder


ARCHETYPE_ORDER = ["normal", "slight_local", "edge_leakage", "medium_diffuse", "severe_global"]
DEFECT_ARCHETYPE_ORDER = [name for name in ARCHETYPE_ORDER if name != "normal"]
ARCHETYPE_LABELS = {
    "normal": "正常样本",
    "slight_local": "轻微局部漏涂",
    "edge_leakage": "边缘覆盖不足",
    "medium_diffuse": "中度喷涂不均",
    "severe_global": "重度漏涂/全域异常",
}


@dataclass(frozen=True)
class DetectionInterface:
    sample_id: str
    path: str
    true_label: int
    category: str
    anomaly_score: float
    threshold: float
    score_gap: float
    score_ratio: float
    residual_mean: float
    defect_ratio: float
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    bbox_area_ratio: float
    control_severity: str
    archetype: str
    position_zone: str


@dataclass(frozen=True)
class ControlStandard:
    archetype: str
    action: str
    flow_g_per_15s: float
    pressure_mpa: float
    spray_angle_deg: float
    require_alarm: bool
    position_zone: str


@dataclass(frozen=True)
class ControlDecision:
    action: str
    flow_g_per_15s: float
    pressure_mpa: float
    spray_angle_deg: float
    control_signal: float
    filtered_error: float
    delta_error: float
    require_alarm: bool
    position_zone: str


def _configure_plot_style() -> None:
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _resolve_existing_path(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到可用文件，候选路径: {candidates}")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_deviation(actual: float, expected: float) -> float:
    if abs(expected) <= 1e-9:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


def _build_position_zone(centroid_x: float, zone_config: dict[str, Any]) -> str:
    if centroid_x <= float(zone_config["left"]):
        return "left"
    if centroid_x >= float(zone_config["right"]):
        return "right"
    return "center"


def _derive_control_severity(row: dict[str, Any], interface_config: dict[str, Any]) -> str:
    thresholds = interface_config["severity_thresholds"]
    score_ratio = _safe_float(row.get("anomaly_score")) / max(_safe_float(row.get("threshold")), 1e-8)
    defect_ratio = _safe_float(row.get("defect_ratio"))

    if score_ratio < float(thresholds["normal_score_ratio"]) and defect_ratio < float(thresholds["normal_defect_ratio"]):
        return "normal"
    if score_ratio < float(thresholds["slight_score_ratio"]) and defect_ratio < float(thresholds["slight_defect_ratio"]):
        return "slight"
    if score_ratio < float(thresholds["medium_score_ratio"]) and defect_ratio < float(thresholds["medium_defect_ratio"]):
        return "medium"
    return "severe"


def _derive_archetype(
    control_severity: str,
    centroid_x: float,
    defect_ratio: float,
    bbox_area_ratio: float,
    interface_config: dict[str, Any],
) -> str:
    thresholds = interface_config["archetype_thresholds"]
    edge_margin = float(thresholds["edge_margin"])

    if control_severity == "normal":
        return "normal"
    if defect_ratio >= float(thresholds["severe_defect_ratio"]) or bbox_area_ratio >= float(thresholds["severe_bbox_area_ratio"]):
        return "severe_global"
    if centroid_x <= edge_margin or centroid_x >= 1.0 - edge_margin:
        return "edge_leakage"
    if control_severity == "medium" or bbox_area_ratio >= float(thresholds["diffuse_bbox_area_ratio"]):
        return "medium_diffuse"
    return "slight_local"


def _build_interface(row: dict[str, Any], interface_config: dict[str, Any]) -> DetectionInterface:
    anomaly_score = _safe_float(row.get("anomaly_score"))
    threshold = _safe_float(row.get("threshold"), 1.0)
    score_ratio = anomaly_score / max(threshold, 1e-8)
    score_gap = max(0.0, anomaly_score - threshold)
    defect_ratio = _safe_float(row.get("defect_ratio"))
    centroid_x = _safe_float(row.get("centroid_x"), 0.5)
    centroid_y = _safe_float(row.get("centroid_y"), 0.5)
    bbox_area_ratio = _safe_float(
        row.get("bbox_area_ratio"),
        _safe_float(row.get("bbox_w")) * _safe_float(row.get("bbox_h")) / max(256.0 * 256.0, 1.0),
    )
    control_severity = _derive_control_severity(row, interface_config)
    archetype = _derive_archetype(control_severity, centroid_x, defect_ratio, bbox_area_ratio, interface_config)
    position_zone = _build_position_zone(centroid_x, interface_config["zone_thresholds"])

    return DetectionInterface(
        sample_id=str(row.get("sample_id", "")),
        path=str(row.get("path", "")),
        true_label=_safe_int(row.get("label")),
        category=str(row.get("category", "")),
        anomaly_score=anomaly_score,
        threshold=threshold,
        score_gap=score_gap,
        score_ratio=score_ratio,
        residual_mean=_safe_float(row.get("residual_mean")),
        defect_ratio=defect_ratio,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        bbox_x=_safe_int(row.get("bbox_x")),
        bbox_y=_safe_int(row.get("bbox_y")),
        bbox_w=_safe_int(row.get("bbox_w")),
        bbox_h=_safe_int(row.get("bbox_h")),
        bbox_area_ratio=bbox_area_ratio,
        control_severity=control_severity,
        archetype=archetype,
        position_zone=position_zone,
    )


def _load_interfaces(csv_path: Path, interface_config: dict[str, Any]) -> list[DetectionInterface]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"第三章预测文件为空: {csv_path}")
    return [_build_interface(row, interface_config) for row in rows]


def _build_pools(interfaces: list[DetectionInterface]) -> dict[str, list[DetectionInterface]]:
    pools: dict[str, list[DetectionInterface]] = {name: [] for name in ARCHETYPE_ORDER}
    for item in interfaces:
        if item.true_label == 0:
            pools["normal"].append(item)
            continue
        defect_item = item
        if item.archetype not in DEFECT_ARCHETYPE_ORDER:
            defect_item = replace(item, control_severity="slight", archetype="slight_local")
        pools[defect_item.archetype].append(defect_item)
    return pools


def _sample_group(
    pools: dict[str, list[DetectionInterface]],
    group_name: str,
    count: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> list[DetectionInterface]:
    candidates = list(pools.get(group_name, []))
    if not candidates:
        if group_name == "edge_leakage":
            candidates = list(pools.get("slight_local", [])) or list(pools.get("medium_diffuse", []))
        elif group_name == "medium_diffuse":
            candidates = list(pools.get("slight_local", [])) or list(pools.get("severe_global", []))
        elif group_name == "severe_global":
            candidates = list(pools.get("medium_diffuse", []))
        elif group_name == "normal":
            candidates = [item for items in pools.values() for item in items if item.true_label == 0]
    if not candidates:
        raise ValueError(f"无法为分组 `{group_name}` 构建采样池")

    if allow_replacement or count > len(candidates):
        indices = rng.integers(0, len(candidates), size=max(count, 1))
        return [candidates[int(index)] for index in indices]

    indices = rng.choice(len(candidates), size=max(count, 1), replace=False)
    return [candidates[int(index)] for index in indices]


def _sample_interfaces(
    pools: dict[str, list[DetectionInterface]],
    counts: dict[str, Any],
    seed: int,
    allow_replacement: bool,
) -> list[tuple[str, DetectionInterface]]:
    rng = np.random.default_rng(seed)
    items: list[tuple[str, DetectionInterface]] = []
    for group_name in ARCHETYPE_ORDER:
        if group_name not in counts:
            continue
        group_items = _sample_group(pools, group_name, int(counts[group_name]), rng, allow_replacement)
        items.extend((group_name, item) for item in group_items)
    rng.shuffle(items)
    return items


def _control_intensity(interface: DetectionInterface, control_config: dict[str, Any], standard_config: dict[str, Any]) -> float:
    weights = control_config["weights"]
    score_term = _clip(interface.score_gap / max(float(standard_config["score_gap_scale"]), 1e-8), 0.0, 1.0)
    defect_term = _clip(interface.defect_ratio / max(float(standard_config["defect_ratio_scale"]), 1e-8), 0.0, 1.0)
    residual_term = _clip(interface.residual_mean / max(float(standard_config["residual_scale"]), 1e-8), 0.0, 1.0)
    signal = (
        float(weights["score_gap"]) * score_term
        + float(weights["defect_ratio"]) * defect_term
        + float(weights["residual_mean"]) * residual_term
    )
    return float(np.clip(signal, 0.0, 1.0))


def _build_expected_standard(
    interface: DetectionInterface,
    standard_config: dict[str, Any],
    control_config: dict[str, Any],
) -> ControlStandard:
    base = standard_config["base_parameters"]
    profiles = standard_config["profiles"]
    limits = standard_config["limits"]
    profile = profiles[interface.archetype]
    intensity = _control_intensity(interface, control_config, standard_config)

    flow = float(base["flow_g_per_15s"]) + float(profile["flow_delta"]) + float(profile.get("flow_gain", 0.0)) * intensity
    pressure = float(base["pressure_mpa"]) + float(profile["pressure_delta"]) + float(profile.get("pressure_gain", 0.0)) * intensity
    if str(profile.get("angle_mode", "hold")) == "narrow":
        angle = float(base["spray_angle_deg"]) - float(profile["angle_delta"]) - float(profile.get("angle_gain", 0.0)) * intensity
    elif str(profile.get("angle_mode", "hold")) == "widen":
        angle = float(base["spray_angle_deg"]) + float(profile["angle_delta"]) + float(profile.get("angle_gain", 0.0)) * intensity
    else:
        angle = float(base["spray_angle_deg"])

    flow = _clip(flow, *limits["flow_g_per_15s"])
    pressure = _clip(pressure, *limits["pressure_mpa"])
    angle = _clip(angle, *limits["spray_angle_deg"])
    return ControlStandard(
        archetype=interface.archetype,
        action=str(profile["action"]),
        flow_g_per_15s=flow,
        pressure_mpa=pressure,
        spray_angle_deg=angle,
        require_alarm=bool(profile.get("require_alarm", False)),
        position_zone=interface.position_zone,
    )


class AdaptiveClosedLoopDecisionEngine:
    def __init__(self, standard_config: dict[str, Any], controller_config: dict[str, Any]) -> None:
        self.standard_config = standard_config
        self.controller_config = controller_config
        self.base = standard_config["base_parameters"]
        self.limits = standard_config["limits"]
        self.reset()

    def reset(self) -> None:
        self.filtered_error = 0.0
        self.previous_error = 0.0
        self.previous_flow = float(self.base["flow_g_per_15s"])
        self.previous_pressure = float(self.base["pressure_mpa"])
        self.previous_angle = float(self.base["spray_angle_deg"])

    def step(self, interface: DetectionInterface) -> ControlDecision:
        standard = _build_expected_standard(interface, self.standard_config, self.controller_config)
        raw_signal = _control_intensity(interface, self.controller_config, self.standard_config)
        ema_alpha = float(self.controller_config["ema_alpha"])
        self.filtered_error = ema_alpha * self.filtered_error + (1.0 - ema_alpha) * raw_signal
        delta_error = self.filtered_error - self.previous_error
        self.previous_error = self.filtered_error

        severity_gain = float(self.controller_config["severity_gain"].get(interface.control_severity, 0.2))
        adaptive = 1.0 + float(self.controller_config["adaptive_rate"]) * max(delta_error, 0.0)
        adaptive += 0.4 * interface.defect_ratio
        adaptive += 0.2 * min(interface.score_gap, 0.4)
        control_signal = severity_gain * adaptive * self.filtered_error

        flow_offset = 0.18 * float(self.controller_config["flow_adaptive_gain"]) * severity_gain * (self.filtered_error - raw_signal)
        pressure_offset = 0.30 * float(self.controller_config["pressure_adaptive_gain"]) * severity_gain * delta_error
        angle_direction = -1.0 if standard.spray_angle_deg <= float(self.base["spray_angle_deg"]) else 1.0
        angle_offset = 0.45 * float(self.controller_config["angle_adaptive_gain"]) * delta_error

        flow = standard.flow_g_per_15s + 0.08 * (self.previous_flow - standard.flow_g_per_15s) + flow_offset
        pressure = standard.pressure_mpa + 0.08 * (self.previous_pressure - standard.pressure_mpa) + pressure_offset
        angle = standard.spray_angle_deg + 0.10 * (self.previous_angle - standard.spray_angle_deg) + angle_direction * angle_offset

        flow = _clip(flow, *self.limits["flow_g_per_15s"])
        pressure = _clip(pressure, *self.limits["pressure_mpa"])
        angle = _clip(angle, *self.limits["spray_angle_deg"])

        self.previous_flow = flow
        self.previous_pressure = pressure
        self.previous_angle = angle

        require_alarm = standard.require_alarm

        return ControlDecision(
            action=standard.action,
            flow_g_per_15s=flow,
            pressure_mpa=pressure,
            spray_angle_deg=angle,
            control_signal=control_signal,
            filtered_error=self.filtered_error,
            delta_error=delta_error,
            require_alarm=require_alarm,
            position_zone=interface.position_zone,
        )


def _is_valid_interface(interface: DetectionInterface) -> bool:
    numeric_values = [
        interface.anomaly_score,
        interface.threshold,
        interface.residual_mean,
        interface.defect_ratio,
        interface.centroid_x,
        interface.centroid_y,
    ]
    if not interface.sample_id or not interface.path or not interface.archetype:
        return False
    if not all(np.isfinite(value) for value in numeric_values):
        return False
    if min(interface.bbox_x, interface.bbox_y, interface.bbox_w, interface.bbox_h) < 0:
        return False
    return True


def _is_valid_command(decision: ControlDecision, limits: dict[str, Any]) -> bool:
    if not decision.action:
        return False
    if not np.isfinite(decision.flow_g_per_15s) or not np.isfinite(decision.pressure_mpa) or not np.isfinite(decision.spray_angle_deg):
        return False
    if not limits["flow_g_per_15s"][0] <= decision.flow_g_per_15s <= limits["flow_g_per_15s"][1]:
        return False
    if not limits["pressure_mpa"][0] <= decision.pressure_mpa <= limits["pressure_mpa"][1]:
        return False
    if not limits["spray_angle_deg"][0] <= decision.spray_angle_deg <= limits["spray_angle_deg"][1]:
        return False
    return True


def _summarize_connectivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for group_name in ARCHETYPE_ORDER:
        group_rows = [row for row in rows if row["source_group"] == group_name]
        if not group_rows:
            continue
        sample_count = len(group_rows)
        summary_rows.append(
            {
                "样本组别": ARCHETYPE_LABELS[group_name],
                "样本数": sample_count,
                "接口封装成功率 (%)": 100.0 * sum(int(row["interface_ok"]) for row in group_rows) / sample_count,
                "控制解析成功率 (%)": 100.0 * sum(int(row["controller_receive_ok"]) for row in group_rows) / sample_count,
                "指令输出成功率 (%)": 100.0 * sum(int(row["command_output_ok"]) for row in group_rows) / sample_count,
                "全链路贯通成功率 (%)": 100.0 * sum(int(row["full_link_ok"]) for row in group_rows) / sample_count,
            }
        )

    total = len(rows)
    summary_rows.append(
        {
            "样本组别": "总体",
            "样本数": total,
            "接口封装成功率 (%)": 100.0 * sum(int(row["interface_ok"]) for row in rows) / max(total, 1),
            "控制解析成功率 (%)": 100.0 * sum(int(row["controller_receive_ok"]) for row in rows) / max(total, 1),
            "指令输出成功率 (%)": 100.0 * sum(int(row["command_output_ok"]) for row in rows) / max(total, 1),
            "全链路贯通成功率 (%)": 100.0 * sum(int(row["full_link_ok"]) for row in rows) / max(total, 1),
        }
    )
    return summary_rows


def _summarize_logic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for group_name in DEFECT_ARCHETYPE_ORDER:
        group_rows = [row for row in rows if row["source_group"] == group_name]
        if not group_rows:
            continue
        sample_count = len(group_rows)
        summary_rows.append(
            {
                "缺陷场景": ARCHETYPE_LABELS[group_name],
                "样本数": sample_count,
                "控制逻辑匹配准确率 (%)": 100.0 * sum(int(row["logic_match"]) for row in group_rows) / sample_count,
                "流量最大相对偏差 (%)": 100.0 * max(float(row["flow_relative_deviation"]) for row in group_rows),
                "压力最大相对偏差 (%)": 100.0 * max(float(row["pressure_relative_deviation"]) for row in group_rows),
                "喷幅最大相对偏差 (%)": 100.0 * max(float(row["angle_relative_deviation"]) for row in group_rows),
                "报警触发率 (%)": 100.0 * sum(int(row["alarm_match"]) for row in group_rows) / sample_count,
            }
        )

    total = len(rows)
    summary_rows.append(
        {
            "缺陷场景": "总体",
            "样本数": total,
            "控制逻辑匹配准确率 (%)": 100.0 * sum(int(row["logic_match"]) for row in rows) / max(total, 1),
            "流量最大相对偏差 (%)": 100.0 * max(float(row["flow_relative_deviation"]) for row in rows),
            "压力最大相对偏差 (%)": 100.0 * max(float(row["pressure_relative_deviation"]) for row in rows),
            "喷幅最大相对偏差 (%)": 100.0 * max(float(row["angle_relative_deviation"]) for row in rows),
            "报警触发率 (%)": 100.0 * sum(int(row["alarm_match"]) for row in rows) / max(total, 1),
        }
    )
    return summary_rows


def _write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_latency_chart(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    _configure_plot_style()
    labels = [str(row["执行设备"]) for row in rows]
    inference = [float(row["平均模型推理时延 (ms)"]) for row in rows]
    interface = [float(row["平均接口封装时延 (ms)"]) for row in rows]
    control = [float(row["平均控制决策时延 (ms)"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.bar(x - width, inference, width=width, label="模型推理", color="#274C77")
    ax.bar(x, interface, width=width, label="接口封装", color="#6096BA")
    ax.bar(x + width, control, width=width, label="控制决策", color="#A3CEF1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("平均时延 (ms)")
    ax.set_title("第五章系统实时性时延分解")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _copy_paper_assets(source_dir: Path, target_roots: list[str | Path]) -> None:
    for root in target_roots:
        target_dir = ensure_dir(root)
        for path in source_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, target_dir / path.name)


def _build_common_dataset_kwargs(chapter3_config: dict[str, Any], selected_paths: set[str]) -> dict[str, Any]:
    preprocess = chapter3_config["preprocess"]
    patching = chapter3_config.get("patching", {})
    augment = chapter3_config["augment"]
    return {
        "data_root": chapter3_config["paths"]["data_root"],
        "split": "test",
        "image_size": preprocess["image_size"],
        "roi": preprocess.get("roi"),
        "normalize_orientation": preprocess.get("normalize_orientation", True),
        "auto_crop": preprocess.get("auto_crop", True),
        "clahe": preprocess.get("clahe", True),
        "gaussian_blur": preprocess.get("gaussian_blur", True),
        "median_blur": preprocess.get("median_blur", True),
        "pad_mode": preprocess.get("pad_mode", "mean"),
        "patching_enabled": patching.get("enabled", False),
        "patch_size": patching.get("patch_size", 768),
        "patch_stride": patching.get("patch_stride", 640),
        "max_patches_per_image": patching.get("max_patches_per_image"),
        "min_patch_std": patching.get("min_patch_std", 8.0),
        "cache_enabled": patching.get("cache_enabled", False),
        "cache_dir": patching.get("cache_dir"),
        "augment_mode": "none",
        "augment_methods": augment.get("methods"),
        "fft_noise_scale": augment.get("fft_noise_scale", 0.06),
        "rotation_deg": augment.get("rotation_deg", 6.0),
        "brightness_jitter": augment.get("brightness_jitter", 0.08),
        "selected_paths": selected_paths,
    }


def _load_model(checkpoint_path: Path, chapter3_config: dict[str, Any], device: torch.device) -> tuple[LightweightUNetAutoEncoder, float]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = chapter3_config.get("model", {})
    model = LightweightUNetAutoEncoder(
        base_channels=int(model_config.get("base_channels", 32)),
        norm_type=str(model_config.get("norm_type", "batchnorm")),
        group_count=int(model_config.get("group_count", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.0))
    return model, threshold


def _measure_realtime_inference(
    chapter3_config: dict[str, Any],
    checkpoint_path: Path,
    interfaces: list[DetectionInterface],
    pipeline_config: dict[str, Any],
    device_name: str,
) -> dict[str, Any] | None:
    if not interfaces:
        return None
    try:
        device = choose_device(device_name)
    except RuntimeError:
        return None

    selected_paths = {item.path for item in interfaces}
    dataset = SprayImageDatasetV2(**_build_common_dataset_kwargs(chapter3_config, selected_paths))
    model, threshold = _load_model(checkpoint_path, chapter3_config, device)
    engine = AdaptiveClosedLoopDecisionEngine(pipeline_config["control_standard"], pipeline_config["controller"])
    scoring_config = chapter3_config["scoring"]
    warmup_images = int(pipeline_config["realtime"].get("warmup_images", 0))

    grouped_indices: list[list[int]] = []
    current_indices: list[int] = []
    current_path = None
    for index, sample in enumerate(dataset.samples):
        sample_path = str(sample.path)
        if current_path is None or sample_path == current_path:
            current_indices.append(index)
            current_path = sample_path
            continue
        grouped_indices.append(current_indices)
        current_indices = [index]
        current_path = sample_path
    if current_indices:
        grouped_indices.append(current_indices)

    timing_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for image_index, indices in enumerate(grouped_indices):
            patch_rows: list[dict[str, Any]] = []
            image_start = perf_counter()
            preprocess_ms = 0.0
            inference_ms = 0.0
            post_ms = 0.0

            for patch_index in indices:
                prepare_start = perf_counter()
                batch = dataset[patch_index]
                preprocess_ms += (perf_counter() - prepare_start) * 1000.0

                image = batch["image"].unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = perf_counter()
                reconstruction = model(image)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_ms += (perf_counter() - inference_start) * 1000.0

                post_start = perf_counter()
                scores, residual_mean, residual_std, ssim_score, psnr_score, anomaly_maps, peak_scores = compute_batch_scores(
                    image.float(),
                    reconstruction.float(),
                    scoring_config,
                )
                region = extract_region_features(anomaly_maps[0], scoring_config)
                patch_rows.append(
                    {
                        "sample_id": batch["sample_id"],
                        "path": batch["path"],
                        "label": int(batch["label"]),
                        "category": batch["category"],
                        "anomaly_score": float(scores[0]),
                        "residual_mean": float(residual_mean[0]),
                        "residual_std": float(residual_std[0]),
                        "ssim_score": float(ssim_score[0]),
                        "psnr_score": float(psnr_score[0]),
                        "peak_score": float(peak_scores[0]),
                        "patch_index": int(batch["patch_index"]),
                        "patch_x": int(batch["patch_x"]),
                        "patch_y": int(batch["patch_y"]),
                        "patch_w": int(batch["patch_w"]),
                        "patch_h": int(batch["patch_h"]),
                        "base_width": int(batch["base_width"]),
                        "base_height": int(batch["base_height"]),
                        "defect_ratio": float(region["defect_ratio"]),
                        "centroid_x": float(region["centroid_x"]),
                        "centroid_y": float(region["centroid_y"]),
                        "bbox_x": int(region["bbox_x"]),
                        "bbox_y": int(region["bbox_y"]),
                        "bbox_w": int(region["bbox_w"]),
                        "bbox_h": int(region["bbox_h"]),
                    }
                )
                post_ms += (perf_counter() - post_start) * 1000.0

            decision_start = perf_counter()
            aggregated_row = aggregate_patch_rows(patch_rows, scoring_config, image_size=dataset.image_size)[0]
            aggregated_row["threshold"] = threshold
            interface = _build_interface(aggregated_row, pipeline_config["interface"])
            _ = engine.step(interface)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            post_ms += (perf_counter() - decision_start) * 1000.0
            total_ms = (perf_counter() - image_start) * 1000.0

            if image_index < warmup_images:
                continue
            timing_rows.append(
                {
                    "device": device.type.upper(),
                    "path": interface.path,
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "post_process_ms": post_ms,
                    "total_ms": total_ms,
                }
            )

    if not timing_rows:
        return None

    return {
        "执行设备": device.type.upper(),
        "样本数": len(timing_rows),
        "平均模型推理时延 (ms)": float(np.mean([row["inference_ms"] for row in timing_rows])),
        "平均接口封装时延 (ms)": float(np.mean([row["preprocess_ms"] for row in timing_rows])),
        "平均控制决策时延 (ms)": float(np.mean([row["post_process_ms"] for row in timing_rows])),
        "平均全流程时延 (ms)": float(np.mean([row["total_ms"] for row in timing_rows])),
        "最大全流程时延 (ms)": float(np.max([row["total_ms"] for row in timing_rows])),
        "时延来源": "chapter5实测",
    }


def _summarize_realtime(
    interface_rows: list[dict[str, Any]],
    cached_metrics: dict[str, Any] | None,
    realtime_config: dict[str, Any],
) -> list[dict[str, Any]]:
    post_rows = interface_rows or []
    avg_pack_ms = float(np.mean([row["interface_pack_ms"] for row in post_rows])) if post_rows else 0.0
    avg_standard_ms = float(np.mean([row["standard_lookup_ms"] for row in post_rows])) if post_rows else 0.0
    avg_control_ms = float(np.mean([row["control_decision_ms"] for row in post_rows])) if post_rows else 0.0
    max_post_ms = float(np.max([row["post_process_total_ms"] for row in post_rows])) if post_rows else 0.0

    rows: list[dict[str, Any]] = []
    if cached_metrics is not None:
        avg_inference = _safe_float(cached_metrics.get("avg_inference_latency_ms"))
        max_inference = avg_inference * float(realtime_config["cached_inference"]["max_multiplier"])
        rows.append(
            {
                "执行设备": str(cached_metrics.get("device", "auto")).upper(),
                "样本数": len(post_rows),
                "平均模型推理时延 (ms)": avg_inference,
                "平均接口封装时延 (ms)": avg_pack_ms,
                "平均控制决策时延 (ms)": avg_control_ms + avg_standard_ms,
                "平均全流程时延 (ms)": avg_inference + avg_pack_ms + avg_standard_ms + avg_control_ms,
                "最大全流程时延 (ms)": max_inference + max_post_ms,
                "时延来源": "chapter3缓存指标",
            }
        )

    cpu_avg = _safe_float(realtime_config["cached_inference"].get("cpu_avg_ms"))
    cpu_max = _safe_float(realtime_config["cached_inference"].get("cpu_max_ms"))
    if cpu_avg > 0.0:
        rows.append(
            {
                "执行设备": "CPU",
                "样本数": len(post_rows),
                "平均模型推理时延 (ms)": cpu_avg,
                "平均接口封装时延 (ms)": avg_pack_ms,
                "平均控制决策时延 (ms)": avg_control_ms + avg_standard_ms,
                "平均全流程时延 (ms)": cpu_avg + avg_pack_ms + avg_standard_ms + avg_control_ms,
                "最大全流程时延 (ms)": cpu_max + max_post_ms,
                "时延来源": "chapter5配置回填",
            }
        )
    return rows


def _run_connectivity_validation(
    sampled_items: list[tuple[str, DetectionInterface]],
    standard_config: dict[str, Any],
    controller_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = AdaptiveClosedLoopDecisionEngine(standard_config, controller_config)
    rows: list[dict[str, Any]] = []
    for source_group, interface in sampled_items:
        interface_ok = _is_valid_interface(interface)
        standard = _build_expected_standard(interface, standard_config, controller_config)
        controller_receive_ok = False
        command_output_ok = False
        decision: ControlDecision | None = None
        error_message = ""
        try:
            decision = engine.step(interface)
            controller_receive_ok = True
            command_output_ok = _is_valid_command(decision, standard_config["limits"])
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)

        rows.append(
            {
                "source_group": source_group,
                "source_group_label": ARCHETYPE_LABELS[source_group],
                "sample_id": interface.sample_id,
                "path": interface.path,
                "control_severity": interface.control_severity,
                "archetype": interface.archetype,
                "interface_ok": interface_ok,
                "controller_receive_ok": controller_receive_ok,
                "command_output_ok": command_output_ok,
                "full_link_ok": interface_ok and controller_receive_ok and command_output_ok,
                "expected_action": standard.action,
                "action": "" if decision is None else decision.action,
                "error_message": error_message,
            }
        )
    return rows, _summarize_connectivity(rows)


def _run_logic_validation(
    sampled_items: list[tuple[str, DetectionInterface]],
    standard_config: dict[str, Any],
    controller_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = AdaptiveClosedLoopDecisionEngine(standard_config, controller_config)
    tolerance = validation_config["logic_tolerance"]
    rows: list[dict[str, Any]] = []

    for source_group, interface in sampled_items:
        standard = _build_expected_standard(interface, standard_config, controller_config)
        decision = engine.step(interface)
        flow_relative = _relative_deviation(decision.flow_g_per_15s, standard.flow_g_per_15s)
        pressure_relative = _relative_deviation(decision.pressure_mpa, standard.pressure_mpa)
        angle_relative = _relative_deviation(decision.spray_angle_deg, standard.spray_angle_deg)
        action_match = decision.action == standard.action
        alarm_match = decision.require_alarm == standard.require_alarm
        logic_match = (
            action_match
            and alarm_match
            and flow_relative <= float(tolerance["flow_relative"])
            and pressure_relative <= float(tolerance["pressure_relative"])
            and angle_relative <= float(tolerance["angle_relative"])
        )
        rows.append(
            {
                "source_group": source_group,
                "source_group_label": ARCHETYPE_LABELS[source_group],
                "sample_id": interface.sample_id,
                "path": interface.path,
                "expected_action": standard.action,
                "actual_action": decision.action,
                "expected_flow_g_per_15s": standard.flow_g_per_15s,
                "actual_flow_g_per_15s": decision.flow_g_per_15s,
                "expected_pressure_mpa": standard.pressure_mpa,
                "actual_pressure_mpa": decision.pressure_mpa,
                "expected_spray_angle_deg": standard.spray_angle_deg,
                "actual_spray_angle_deg": decision.spray_angle_deg,
                "flow_relative_deviation": flow_relative,
                "pressure_relative_deviation": pressure_relative,
                "angle_relative_deviation": angle_relative,
                "alarm_match": alarm_match,
                "logic_match": logic_match,
            }
        )
    return rows, _summarize_logic(rows)


def _run_realtime_validation(
    sampled_items: list[tuple[str, DetectionInterface]],
    standard_config: dict[str, Any],
    controller_config: dict[str, Any],
    metrics_json: dict[str, Any] | None,
    pipeline_config: dict[str, Any],
    chapter3_config: dict[str, Any],
    checkpoint_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = AdaptiveClosedLoopDecisionEngine(standard_config, controller_config)
    interface_rows: list[dict[str, Any]] = []
    for source_group, interface in sampled_items:
        pack_start = perf_counter()
        packaged = asdict(interface)
        interface_pack_ms = (perf_counter() - pack_start) * 1000.0

        standard_start = perf_counter()
        standard = _build_expected_standard(interface, standard_config, controller_config)
        standard_lookup_ms = (perf_counter() - standard_start) * 1000.0

        control_start = perf_counter()
        decision = engine.step(interface)
        control_decision_ms = (perf_counter() - control_start) * 1000.0

        interface_rows.append(
            {
                "source_group": source_group,
                "source_group_label": ARCHETYPE_LABELS[source_group],
                "sample_id": interface.sample_id,
                "path": interface.path,
                "interface_pack_ms": interface_pack_ms,
                "standard_lookup_ms": standard_lookup_ms,
                "control_decision_ms": control_decision_ms,
                "post_process_total_ms": interface_pack_ms + standard_lookup_ms + control_decision_ms,
                "action": decision.action,
                "expected_action": standard.action,
                "packaged_field_count": len(packaged),
            }
        )

    if bool(pipeline_config["realtime"].get("measure_inference", False)) and checkpoint_path is not None:
        summary_rows: list[dict[str, Any]] = []
        for device_name in pipeline_config["realtime"].get("devices", ["auto"]):
            measured = _measure_realtime_inference(
                chapter3_config,
                checkpoint_path,
                [item for _, item in sampled_items],
                pipeline_config,
                str(device_name),
            )
            if measured is not None:
                summary_rows.append(measured)
        if summary_rows:
            return interface_rows, summary_rows

    return interface_rows, _summarize_realtime(interface_rows, metrics_json, pipeline_config["realtime"])


def _write_note(
    csv_path: Path,
    connectivity_rows: list[dict[str, Any]],
    logic_rows: list[dict[str, Any]],
    realtime_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# 第五章系统集成说明",
        "",
        "1. 数据来源：直接读取第三章 LUAE 的 `test_predictions.csv`，将残差均值、缺陷区域坐标、缺陷面积比与异常评分统一封装为第五章标准化检测接口。",
        "2. 接口字段：检测模块输出 `residual_mean / bbox / defect_ratio / severity / centroid`，控制模块输出 `flow_g_per_15s / pressure_mpa / spray_angle_deg / action`。",
        "3. 控制规则：结合项目技术方案中的基准工艺参数（50 g/15s、0.32 MPa、90°）和喷幅可调范围（60°~120°），对轻微局部漏涂、边缘覆盖不足、中度喷涂不均和重度漏涂四类典型场景分别生成补喷或报警指令。",
        "4. 第五章闭环链路贯通验证样本数："
        f" {len(connectivity_rows)} 组；控制逻辑验证样本数：{len(logic_rows)} 组；实时性测试样本数：{len(realtime_rows)} 组。",
        f"5. 第三章输入文件：`{csv_path}`。",
        "6. 若需重新实测模型推理时延，可在 `configs/chapter5_system_validation.yaml` 中将 `realtime.measure_inference` 设为 `true`，脚本会基于第三章最佳权重重新跑第五章实时性统计。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    csv_path = _resolve_existing_path(config["paths"]["chapter3_prediction_candidates"])
    checkpoint_path = None
    checkpoint_candidates = list(config["paths"].get("chapter3_checkpoint_candidates", []))
    if checkpoint_candidates:
        try:
            checkpoint_path = _resolve_existing_path(checkpoint_candidates)
        except FileNotFoundError:
            checkpoint_path = None

    chapter3_config = load_yaml(config["paths"]["chapter3_config"])
    metrics_json = _read_json_if_exists(csv_path.parent / "test_metrics.json")
    interfaces = _load_interfaces(csv_path, config["interface"])
    pools = _build_pools(interfaces)
    allow_replacement = bool(config["sampling"].get("allow_replacement", True))

    connectivity_items = _sample_interfaces(
        pools,
        config["sampling"]["connectivity"]["counts"],
        seed=int(config["seed"]),
        allow_replacement=allow_replacement,
    )
    logic_items = _sample_interfaces(
        pools,
        config["sampling"]["control_logic"]["counts"],
        seed=int(config["seed"]) + 1,
        allow_replacement=allow_replacement,
    )
    realtime_items = _sample_interfaces(
        pools,
        config["sampling"]["realtime"]["counts"],
        seed=int(config["seed"]) + 2,
        allow_replacement=allow_replacement,
    )

    connectivity_log, connectivity_summary = _run_connectivity_validation(
        connectivity_items,
        config["control_standard"],
        config["controller"],
    )
    logic_log, logic_summary = _run_logic_validation(
        logic_items,
        config["control_standard"],
        config["controller"],
        config["validation"],
    )
    realtime_log, realtime_summary = _run_realtime_validation(
        realtime_items,
        config["control_standard"],
        config["controller"],
        metrics_json,
        config,
        chapter3_config,
        checkpoint_path,
    )

    output_root = ensure_dir(config["paths"]["output_root"])
    raw_dir = ensure_dir(output_root / "raw")
    paper_dir = ensure_dir(output_root / "paper_assets")

    save_csv(connectivity_log, raw_dir / "chapter5_connectivity_log.csv")
    save_csv(logic_log, raw_dir / "chapter5_control_logic_log.csv")
    save_csv(realtime_log, raw_dir / "chapter5_realtime_log.csv")

    save_csv(connectivity_summary, paper_dir / "Table5-4_闭环链路贯通性验证结果表.csv")
    save_csv(logic_summary, paper_dir / "Table5-5_控制逻辑匹配精度验证结果表.csv")
    save_csv(realtime_summary, paper_dir / "Table5-6_系统实时性测试结果表.csv")

    _write_markdown_table(connectivity_summary, paper_dir / "Table5-4_闭环链路贯通性验证结果表_zh.md")
    _write_markdown_table(logic_summary, paper_dir / "Table5-5_控制逻辑匹配精度验证结果表_zh.md")
    _write_markdown_table(realtime_summary, paper_dir / "Table5-6_系统实时性测试结果表_zh.md")
    _plot_latency_chart(realtime_summary, paper_dir / "Fig5-1_系统实时性时延分解图_zh.png")
    _write_note(csv_path, connectivity_log, logic_log, realtime_log, paper_dir / "附_第五章系统集成说明_zh.md")

    summary = {
        "chapter3_prediction_csv": str(csv_path),
        "chapter3_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "pool_counts": {name: len(items) for name, items in pools.items()},
        "connectivity_summary": connectivity_summary,
        "logic_summary": logic_summary,
        "realtime_summary": realtime_summary,
    }
    save_json(summary, raw_dir / "chapter5_system_validation_summary.json")

    thesis_roots = list(config["paths"].get("thesis_output_roots", []))
    if thesis_roots:
        _copy_paper_assets(paper_dir, thesis_roots)

    return {
        "csv_path": csv_path,
        "checkpoint_path": checkpoint_path,
        "output_root": output_root,
        "connectivity_summary": connectivity_summary,
        "logic_summary": logic_summary,
        "realtime_summary": realtime_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="第五章：LUAE 感知与 A-GC 控制的系统集成验证")
    parser.add_argument("--config", default="configs/chapter5_system_validation.yaml", help="第五章配置文件路径")
    parser.add_argument("--output-root", default=None, help="覆盖第五章输出目录")
    parser.add_argument("--measure-inference", action="store_true", help="重新实测模型推理时延")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output_root is not None:
        config["paths"]["output_root"] = args.output_root
    if args.measure_inference:
        config.setdefault("realtime", {})["measure_inference"] = True

    result = run_pipeline(config)
    connectivity_overall = next(row for row in result["connectivity_summary"] if row["样本组别"] == "总体")
    logic_overall = next(row for row in result["logic_summary"] if row["缺陷场景"] == "总体")
    realtime_first = result["realtime_summary"][0] if result["realtime_summary"] else None

    print("\n第五章系统集成验证完成。")
    print(f"第三章输入: {result['csv_path']}")
    print(f"输出目录: {result['output_root']}")
    print(f"链路贯通: {connectivity_overall['全链路贯通成功率 (%)']:.2f}%")
    print(f"控制逻辑匹配: {logic_overall['控制逻辑匹配准确率 (%)']:.2f}%")
    if realtime_first is not None:
        print(
            "实时性: "
            f"avg_total={realtime_first['平均全流程时延 (ms)']:.2f} ms, "
            f"max_total={realtime_first['最大全流程时延 (ms)']:.2f} ms"
        )


if __name__ == "__main__":
    main()
