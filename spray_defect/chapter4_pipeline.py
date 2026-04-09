from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from .config import ensure_dir, load_yaml, save_csv, save_json


CONTROLLER_ORDER = ["PID", "Fuzzy-PID", "MA-GC", "RL-GC", "A-GC"]
BASELINE_CONTROLLER = "PID"
PROPOSED_CONTROLLER = "A-GC"

CONTROLLER_LABELS = {
    "PID": "PID",
    "Fuzzy-PID": "Fuzzy-PID",
    "MA-GC": "MA-GC",
    "RL-GC": "RL-GC",
    "A-GC": "A-GC（本文算法）",
}

CONTROLLER_COLORS = {
    "PID": "#7A7A7A",
    "Fuzzy-PID": "#5E8C61",
    "MA-GC": "#A1794A",
    "RL-GC": "#9C6644",
    "A-GC": "#274C77",
}

METRIC_SPECS = [
    ("steady_state_error_um", "稳态误差 |e_ss| (μm)", "lower"),
    ("settling_time_s", "调节时间 t_s (s)", "lower"),
    ("uniformity_percent", "涂层均匀度 U_c (%)", "higher"),
    ("perception_error_energy", "感知误差能量 E_p", "lower"),
    ("control_smoothness", "控制平滑度 S_u", "lower"),
    ("overshoot_percent", "超调量 M_p (%)", "lower"),
]

SCORE_METRICS = [metric_key for metric_key, _, _ in METRIC_SPECS if metric_key != "overshoot_percent"]


@dataclass(frozen=True)
class VisualSignal:
    sample_id: str
    path: str
    category: str
    severity: str
    residual_mean: float
    anomaly_score: float
    threshold: float
    defect_ratio: float
    centroid_x: float
    centroid_y: float
    score_gap: float
    raw_error: float

    @property
    def spatial_error(self) -> float:
        return self.centroid_x - 0.5


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    severity: str
    count: int
    kind: str


@dataclass(frozen=True)
class ControlObservation:
    cycle: int
    phase_name: str
    severity: str
    raw_error: float
    measured_error: float
    score_gap: float
    defect_ratio: float
    spatial_error: float
    target_thickness_um: float
    thickness_um: float

    @property
    def thickness_error_um(self) -> float:
        return self.target_thickness_um - self.thickness_um


@dataclass(frozen=True)
class ControlCommand:
    flow_g_per_15s: float
    pressure_mpa: float
    spray_angle_deg: float
    control_signal: float
    filtered_error: float
    delta_error: float
    action: str


def _configure_plot_style() -> None:
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum - minimum <= 1e-9:
        return 0.0
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _resolve_existing_path(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到可用的第三章 LUAE 输出文件，候选路径: {candidates}")


def _load_visual_signal_library(csv_path: Path, config: dict[str, Any]) -> tuple[dict[str, list[VisualSignal]], dict[str, Any]]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"第三章预测文件为空: {csv_path}")

    gaps = [max(0.0, _safe_float(row.get("anomaly_score")) - _safe_float(row.get("threshold"))) for row in rows]
    residuals = [_safe_float(row.get("residual_mean")) for row in rows]
    defect_ratios = [_safe_float(row.get("defect_ratio")) for row in rows]

    residual_min, residual_max = min(residuals), max(residuals)
    gap_min, gap_max = min(gaps), max(gaps)
    defect_min, defect_max = min(defect_ratios), max(defect_ratios)

    weights = config["visual_error"]
    library: dict[str, list[VisualSignal]] = defaultdict(list)
    for row in rows:
        severity = str(row.get("severity", "normal")).strip() or "normal"
        residual_mean = _safe_float(row.get("residual_mean"))
        anomaly_score = _safe_float(row.get("anomaly_score"))
        threshold = _safe_float(row.get("threshold"))
        defect_ratio = _safe_float(row.get("defect_ratio"))
        score_gap = max(0.0, anomaly_score - threshold)

        residual_norm = _normalize(residual_mean, residual_min, residual_max)
        gap_norm = _normalize(score_gap, gap_min, gap_max)
        defect_norm = _normalize(defect_ratio, defect_min, defect_max)

        raw_error = (
            float(weights["residual_weight"]) * residual_norm
            + float(weights["anomaly_gap_weight"]) * gap_norm
            + float(weights["defect_ratio_weight"]) * defect_norm
        )
        if severity == "normal":
            raw_error *= 0.92
        elif severity == "severe":
            raw_error = min(1.0, raw_error + 0.06)

        library[severity].append(
            VisualSignal(
                sample_id=str(row.get("sample_id", "")),
                path=str(row.get("path", "")),
                category=str(row.get("category", "")),
                severity=severity,
                residual_mean=residual_mean,
                anomaly_score=anomaly_score,
                threshold=threshold,
                defect_ratio=defect_ratio,
                centroid_x=_safe_float(row.get("centroid_x"), 0.5),
                centroid_y=_safe_float(row.get("centroid_y"), 0.5),
                score_gap=score_gap,
                raw_error=float(np.clip(raw_error, 0.0, 1.0)),
            )
        )

    summary = {
        "source_csv": str(csv_path),
        "num_rows": len(rows),
        "severity_counts": {severity: len(signals) for severity, signals in library.items()},
        "raw_error_range": {
            "min": float(min(signal.raw_error for signals in library.values() for signal in signals)),
            "max": float(max(signal.raw_error for signals in library.values() for signal in signals)),
        },
    }
    return library, summary


def _select_sampling_pool(library: dict[str, list[VisualSignal]], phase: PhaseSpec) -> list[VisualSignal]:
    signals = list(library.get(phase.severity, []))
    if not signals:
        raise ValueError(f"缺少严重度 `{phase.severity}` 的 LUAE 残差信号，无法构建阶段 `{phase.name}`")

    ordered = sorted(signals, key=lambda signal: signal.raw_error)
    if phase.kind in {"baseline", "recovery"} and phase.severity == "normal":
        keep = max(1, math.ceil(len(ordered) * 0.75))
        return ordered[:keep]
    if phase.severity == "slight":
        return ordered[max(0, len(ordered) // 5) :]
    if phase.severity == "medium":
        return ordered[max(0, len(ordered) // 4) :]
    return ordered


def _build_trial_sequence(
    library: dict[str, list[VisualSignal]],
    phase_specs: list[PhaseSpec],
    seed: int,
) -> tuple[list[tuple[VisualSignal, PhaseSpec]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    sequence: list[tuple[VisualSignal, PhaseSpec]] = []
    phase_ranges: list[dict[str, Any]] = []
    cursor = 0

    for phase in phase_specs:
        pool = _select_sampling_pool(library, phase)
        indices = rng.integers(0, len(pool), size=max(int(phase.count), 1))
        samples = [pool[int(index)] for index in indices]
        start = cursor
        for sample in samples:
            sequence.append((sample, phase))
            cursor += 1
        phase_ranges.append(
            {
                "name": phase.name,
                "kind": phase.kind,
                "severity": phase.severity,
                "start": start,
                "end": cursor,
            }
        )

    return sequence, phase_ranges


class BaseController:
    def __init__(self, name: str, config: dict[str, Any], simulation_config: dict[str, Any], seed: int) -> None:
        self.name = name
        self.config = config
        self.base = simulation_config["base_parameters"]
        self.limits = simulation_config["limits"]
        self.command_scales = simulation_config["command_scales"]
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.filtered_error = 0.0

    def _compose_command(
        self,
        effort: float,
        observation: ControlObservation,
        *,
        spatial_gain: float,
        action: str,
        bias_flow: float = 0.0,
        bias_pressure: float = 0.0,
        bias_angle: float = 0.0,
    ) -> ControlCommand:
        max_effort = float(self.config.get("max_effort", 1.0))
        effort = float(np.tanh(effort / max(max_effort, 1e-6)) * max_effort)
        delta_error = effort - self.previous_error
        self.previous_error = effort

        spatial_term = spatial_gain * observation.spatial_error
        flow = self.base["flow_g_per_15s"] + bias_flow + self.command_scales["flow_g_per_15s"] * effort
        pressure = self.base["pressure_mpa"] + bias_pressure + self.command_scales["pressure_mpa"] * effort
        angle = (
            self.base["spray_angle_deg"]
            + bias_angle
            - self.command_scales["spray_angle_deg"] * (0.40 * effort + spatial_term)
        )

        flow = _clip(flow, *self.limits["flow_g_per_15s"])
        pressure = _clip(pressure, *self.limits["pressure_mpa"])
        angle = _clip(angle, *self.limits["spray_angle_deg"])

        return ControlCommand(
            flow_g_per_15s=flow,
            pressure_mpa=pressure,
            spray_angle_deg=angle,
            control_signal=effort,
            filtered_error=float(self.filtered_error),
            delta_error=float(delta_error),
            action=action,
        )

    def step(self, observation: ControlObservation) -> ControlCommand:
        raise NotImplementedError


class PIDController(BaseController):
    def step(self, observation: ControlObservation) -> ControlCommand:
        error = observation.measured_error + 0.05 * abs(observation.thickness_error_um) / observation.target_thickness_um
        derivative = error - self.filtered_error
        self.filtered_error = error
        self.integral_error = _clip(
            self.integral_error + error,
            -float(self.config["integral_limit"]),
            float(self.config["integral_limit"]),
        )
        effort = (
            float(self.config["kp"]) * error
            + float(self.config["ki"]) * self.integral_error
            + float(self.config["kd"]) * derivative
        )
        if error < 0.05 and abs(observation.thickness_error_um) < 0.30:
            action = "稳态保持"
        else:
            action = "常规补偿"
        return self._compose_command(
            effort,
            observation,
            spatial_gain=float(self.config.get("spatial_gain", 0.10)),
            action=action,
        )


class FuzzyPIDController(BaseController):
    def step(self, observation: ControlObservation) -> ControlCommand:
        error = observation.measured_error + 0.04 * abs(observation.thickness_error_um) / observation.target_thickness_um
        derivative = error - self.filtered_error
        self.filtered_error = error
        self.integral_error = _clip(
            self.integral_error + error,
            -float(self.config["integral_limit"]),
            float(self.config["integral_limit"]),
        )

        gain_scale = 1.0
        if error >= float(self.config["high_error"]):
            gain_scale += float(self.config["gain_boost"])
        elif error <= float(self.config["low_error"]):
            gain_scale -= float(self.config["gain_drop"])
        if derivative > 0.04:
            gain_scale += 0.15
        elif derivative < -0.04:
            gain_scale -= 0.08

        effort = gain_scale * (
            float(self.config["kp"]) * error
            + float(self.config["ki"]) * self.integral_error
            + float(self.config["kd"]) * derivative
        )
        effort += 0.10 * observation.defect_ratio
        return self._compose_command(
            effort,
            observation,
            spatial_gain=float(self.config.get("spatial_gain", 0.15)),
            action="模糊增益调节",
        )


class MAGainController(BaseController):
    def reset(self) -> None:
        super().reset()
        self.window = deque(maxlen=int(self.config["window"]))
        self.last_average = 0.0

    def step(self, observation: ControlObservation) -> ControlCommand:
        error = observation.measured_error + 0.03 * abs(observation.thickness_error_um) / observation.target_thickness_um
        self.window.append(error)
        moving_average = float(np.mean(self.window))
        derivative = moving_average - self.last_average
        self.last_average = moving_average
        self.filtered_error = moving_average

        effort = (
            float(self.config["gain"]) * moving_average
            + float(self.config["derivative_gain"]) * derivative
            + 0.08 * observation.defect_ratio
        )
        if observation.severity == "normal" and moving_average < 0.06:
            effort *= 0.82
        return self._compose_command(
            effort,
            observation,
            spatial_gain=float(self.config.get("spatial_gain", 0.08)),
            action="滑动均值补偿",
        )


class RLGainController(BaseController):
    def reset(self) -> None:
        super().reset()
        self.policy_gain = float(self.config["init_gain"])
        self.previous_reward_error = 0.0
        self.exploration_scale = float(self.config["exploration_scale"])
        self.step_index = 0

    def step(self, observation: ControlObservation) -> ControlCommand:
        error = observation.measured_error + 0.05 * abs(observation.thickness_error_um) / observation.target_thickness_um
        derivative = error - self.filtered_error
        self.filtered_error = error

        if self.step_index > 0:
            improvement = self.previous_reward_error - error
            reward = improvement - 0.06 * abs(observation.thickness_error_um) / observation.target_thickness_um
            self.policy_gain += float(self.config["learning_rate"]) * reward
            self.policy_gain = _clip(self.policy_gain, *self.config["gain_limits"])

        exploration = self.exploration_scale * math.sin(1.618 * self.step_index) + self.rng.normal(0.0, 0.035)
        self.exploration_scale *= float(self.config["exploration_decay"])
        effort = self.policy_gain * error + 0.12 * max(derivative, 0.0) + exploration

        self.previous_reward_error = error
        self.step_index += 1
        return self._compose_command(
            effort,
            observation,
            spatial_gain=float(self.config.get("spatial_gain", 0.15)),
            action="在线增益探索",
        )


class AdaptiveGainController(BaseController):
    def reset(self) -> None:
        super().reset()
        self.trend_error = 0.0

    def step(self, observation: ControlObservation) -> ControlCommand:
        base_error = 0.72 * observation.measured_error + 0.28 * observation.raw_error
        self.filtered_error = (
            float(self.config["ema_alpha"]) * self.filtered_error
            + (1.0 - float(self.config["ema_alpha"])) * base_error
        )
        derivative = self.filtered_error - self.trend_error
        self.trend_error = self.filtered_error
        self.integral_error = _clip(
            self.integral_error + self.filtered_error,
            -float(self.config["integral_limit"]),
            float(self.config["integral_limit"]),
        )

        severity_gain = float(self.config["severity_gain"].get(observation.severity, 0.2))
        adaptive_factor = (
            1.0
            + float(self.config["adaptive_rate"]) * max(derivative, 0.0)
            + float(self.config["defect_ratio_gain"]) * observation.defect_ratio
            + float(self.config["score_gap_gain"]) * observation.score_gap
        )
        if abs(observation.thickness_error_um) > 0.55:
            adaptive_factor += 0.10 * abs(observation.thickness_error_um)

        effort = severity_gain * adaptive_factor * (
            float(self.config["kp"]) * self.filtered_error
            + float(self.config["ki"]) * self.integral_error
            + float(self.config["kd"]) * derivative
        )

        if observation.severity == "severe":
            bias_flow = 0.65
            bias_pressure = 0.003
            bias_angle = -0.30 * np.sign(observation.spatial_error)
            action = "强缺陷自适应补喷"
        elif observation.severity in {"slight", "medium"}:
            bias_flow = 0.25
            bias_pressure = 0.001
            bias_angle = -0.12 * np.sign(observation.spatial_error)
            action = "局部自适应补偿"
        else:
            bias_flow = 0.0
            bias_pressure = 0.0
            bias_angle = 0.0
            action = "闭环稳态保持"

        if observation.raw_error < 0.05 and abs(observation.thickness_error_um) < 0.25:
            effort *= 0.52
        elif observation.severity == "normal":
            effort *= 0.60

        effort = 0.58 * self.previous_error + 0.42 * effort
        spatial_gain = float(self.config.get("spatial_gain", 0.40))
        if observation.severity == "normal":
            spatial_gain *= 0.25
        elif observation.severity == "slight":
            spatial_gain *= 0.45
        elif observation.severity == "medium":
            spatial_gain *= 0.60
        else:
            spatial_gain *= 0.72

        return self._compose_command(
            effort,
            observation,
            spatial_gain=spatial_gain,
            action=action,
            bias_flow=bias_flow,
            bias_pressure=bias_pressure,
            bias_angle=bias_angle,
        )


def _build_controller(name: str, config: dict[str, Any], simulation_config: dict[str, Any], seed: int) -> BaseController:
    if name == "PID":
        return PIDController(name, config, simulation_config, seed)
    if name == "Fuzzy-PID":
        return FuzzyPIDController(name, config, simulation_config, seed)
    if name == "MA-GC":
        return MAGainController(name, config, simulation_config, seed)
    if name == "RL-GC":
        return RLGainController(name, config, simulation_config, seed)
    if name == "A-GC":
        return AdaptiveGainController(name, config, simulation_config, seed)
    raise ValueError(f"不支持的控制算法: {name}")


class ClosedLoopSprayPlant:
    def __init__(self, simulation_config: dict[str, Any], plant_config: dict[str, Any], seed: int) -> None:
        self.target_thickness_um = float(simulation_config["target_thickness_um"])
        self.thickness_limits_um = plant_config["thickness_limits_um"]
        self.base = simulation_config["base_parameters"]
        self.limits = simulation_config["limits"]
        self.inertia = float(plant_config["inertia"])
        self.control_gain = float(plant_config["control_gain"])
        self.disturbance_gain = float(plant_config["disturbance_gain"])
        self.coupling_gain = float(plant_config["coupling_gain"])
        self.balance_decay = float(plant_config["balance_decay"])
        self.balance_disturbance_gain = float(plant_config["balance_disturbance_gain"])
        self.balance_control_gain = float(plant_config["balance_control_gain"])
        self.thickness_noise_um = float(plant_config["thickness_noise_um"])
        self.balance_noise = float(plant_config["balance_noise"])
        self.measurement_noise = float(plant_config["measurement_noise"])
        self.residual_noise = float(plant_config["residual_noise"])
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.thickness_um = self.target_thickness_um - 0.75
        self.balance = 0.0
        self.effective_error = 0.08
        self.uniformity_percent = 95.0
        self.combined_command = 0.0

    def observe(self, signal: VisualSignal) -> float:
        thickness_component = abs(self.target_thickness_um - self.thickness_um) / max(self.target_thickness_um, 1e-6)
        measured_error = (
            0.64 * signal.raw_error
            + 0.18 * self.effective_error
            + 0.12 * abs(self.balance)
            + 0.06 * thickness_component
        )
        measured_error += self.rng.normal(0.0, self.measurement_noise)
        return max(0.0, float(measured_error))

    def step(self, signal: VisualSignal, command: ControlCommand) -> dict[str, float]:
        flow_norm = (command.flow_g_per_15s - self.base["flow_g_per_15s"]) / max(
            self.limits["flow_g_per_15s"][1] - self.base["flow_g_per_15s"], 1e-8
        )
        pressure_norm = (command.pressure_mpa - self.base["pressure_mpa"]) / max(
            self.limits["pressure_mpa"][1] - self.base["pressure_mpa"], 1e-8
        )
        angle_norm = (self.base["spray_angle_deg"] - command.spray_angle_deg) / max(
            self.base["spray_angle_deg"] - self.limits["spray_angle_deg"][0], 1e-8
        )

        global_control = 0.54 * flow_norm + 0.30 * pressure_norm + 0.16 * angle_norm
        spatial_alignment = angle_norm * np.sign(signal.spatial_error)
        disturbance_level = signal.raw_error * (0.85 + 4.0 * signal.defect_ratio) + 0.35 * signal.score_gap
        disturbance_spatial = signal.spatial_error * (0.55 + 8.0 * signal.defect_ratio)

        next_thickness = (
            self.inertia * self.thickness_um
            + (1.0 - self.inertia) * self.target_thickness_um
            + self.control_gain * global_control
            - self.disturbance_gain * disturbance_level
            - self.coupling_gain * abs(self.balance)
            + self.rng.normal(0.0, self.thickness_noise_um)
        )
        self.thickness_um = _clip(next_thickness, *self.thickness_limits_um)

        next_balance = (
            self.balance_decay * self.balance
            + self.balance_disturbance_gain * disturbance_spatial
            - self.balance_control_gain * spatial_alignment
            + self.rng.normal(0.0, self.balance_noise)
        )
        self.balance = _clip(next_balance, -1.0, 1.0)

        residual_compensation = 0.36 * max(global_control, 0.0) + 0.28 * max(spatial_alignment * np.sign(signal.spatial_error), 0.0)
        self.effective_error = max(
            0.0,
            float(
                signal.raw_error * (1.0 - residual_compensation)
                + 0.05 * abs(self.target_thickness_um - self.thickness_um) / self.target_thickness_um
                + 0.08 * abs(self.balance)
                + self.rng.normal(0.0, self.residual_noise)
            ),
        )

        uniformity = 1.0 - 0.032 * abs(self.target_thickness_um - self.thickness_um) - 0.42 * self.effective_error - 0.18 * abs(self.balance)
        self.uniformity_percent = float(np.clip(100.0 * uniformity, 0.0, 100.0))
        self.combined_command = float(0.46 * flow_norm + 0.29 * pressure_norm + 0.25 * angle_norm)

        return {
            "thickness_um": float(self.thickness_um),
            "thickness_error_um": float(self.target_thickness_um - self.thickness_um),
            "effective_visual_error": float(self.effective_error),
            "uniformity_percent": float(self.uniformity_percent),
            "balance": float(self.balance),
            "combined_command": float(self.combined_command),
        }


def _summarize_trial_metrics(
    log_rows: list[dict[str, Any]],
    phase_ranges: list[dict[str, Any]],
    simulation_config: dict[str, Any],
) -> dict[str, float]:
    target_thickness_um = float(simulation_config["target_thickness_um"])
    cycle_time_s = float(simulation_config["cycle_time_s"])
    settle_band_um = float(simulation_config["settle_band_um"])

    last_recovery = next(phase for phase in reversed(phase_ranges) if phase["kind"] == "recovery")
    steady_segment = log_rows[last_recovery["start"] : last_recovery["end"]]
    steady_state_error = float(np.mean([abs(row["thickness_error_um"]) for row in steady_segment]))

    settling_times: list[float] = []
    for phase in phase_ranges:
        if phase["kind"] != "recovery":
            continue
        settled_offset: int | None = None
        for index in range(phase["start"], phase["end"]):
            if abs(log_rows[index]["thickness_error_um"]) <= settle_band_um:
                settled_offset = index - phase["start"]
                break
        if settled_offset is None:
            settled_offset = phase["end"] - phase["start"]
        settling_times.append(settled_offset * cycle_time_s)

    control_signal = np.asarray([row["combined_command"] for row in log_rows], dtype=np.float64)
    control_deltas = np.diff(control_signal) if control_signal.size >= 2 else np.asarray([0.0], dtype=np.float64)
    overshoot = max(max(0.0, row["thickness_um"] - target_thickness_um) for row in log_rows) / target_thickness_um * 100.0

    return {
        "steady_state_error_um": steady_state_error,
        "settling_time_s": float(np.mean(settling_times)),
        "uniformity_percent": float(np.mean([row["uniformity_percent"] for row in log_rows])),
        "perception_error_energy": float(np.mean([row["effective_visual_error"] ** 2 for row in log_rows])),
        "control_smoothness": float(np.mean(control_deltas**2)),
        "overshoot_percent": float(overshoot),
    }


def _run_single_trial(
    controller_name: str,
    controller_config: dict[str, Any],
    simulation_config: dict[str, Any],
    plant_config: dict[str, Any],
    sequence: list[tuple[VisualSignal, PhaseSpec]],
    phase_ranges: list[dict[str, Any]],
    trial_index: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    controller = _build_controller(controller_name, controller_config, simulation_config, seed)
    plant = ClosedLoopSprayPlant(simulation_config, plant_config, seed)

    log_rows: list[dict[str, Any]] = []
    for cycle, (signal, phase) in enumerate(sequence, start=1):
        measured_error = plant.observe(signal)
        observation = ControlObservation(
            cycle=cycle,
            phase_name=phase.name,
            severity=signal.severity,
            raw_error=signal.raw_error,
            measured_error=measured_error,
            score_gap=signal.score_gap,
            defect_ratio=signal.defect_ratio,
            spatial_error=signal.spatial_error,
            target_thickness_um=plant.target_thickness_um,
            thickness_um=plant.thickness_um,
        )
        command = controller.step(observation)
        state = plant.step(signal, command)
        log_rows.append(
            {
                "controller": controller_name,
                "controller_label": CONTROLLER_LABELS[controller_name],
                "trial": trial_index,
                "cycle": cycle,
                "phase_name": phase.name,
                "phase_kind": phase.kind,
                "severity": signal.severity,
                "sample_id": signal.sample_id,
                "sample_path": signal.path,
                "raw_visual_error": signal.raw_error,
                "measured_error": measured_error,
                "score_gap": signal.score_gap,
                "defect_ratio": signal.defect_ratio,
                "spatial_error": signal.spatial_error,
                "flow_g_per_15s": command.flow_g_per_15s,
                "pressure_mpa": command.pressure_mpa,
                "spray_angle_deg": command.spray_angle_deg,
                "control_signal": command.control_signal,
                "filtered_error": command.filtered_error,
                "delta_error": command.delta_error,
                "action": command.action,
                "thickness_um": state["thickness_um"],
                "thickness_error_um": state["thickness_error_um"],
                "target_thickness_um": observation.target_thickness_um,
                "effective_visual_error": state["effective_visual_error"],
                "uniformity_percent": state["uniformity_percent"],
                "combined_command": state["combined_command"],
                "balance": state["balance"],
            }
        )

    metrics = _summarize_trial_metrics(log_rows, phase_ranges, simulation_config)
    return log_rows, metrics


def _aggregate_summary(trial_rows: list[dict[str, Any]], scoring_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    by_controller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        by_controller[str(row["controller"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    std_lookup: dict[str, dict[str, float]] = {}
    for controller_name in CONTROLLER_ORDER:
        rows = by_controller[controller_name]
        if not rows:
            continue
        summary_row = {
            "controller": controller_name,
            "controller_label": CONTROLLER_LABELS[controller_name],
        }
        std_lookup[controller_name] = {}
        for metric_key, _, _ in METRIC_SPECS:
            values = np.asarray([_safe_float(row.get(metric_key)) for row in rows], dtype=np.float64)
            summary_row[metric_key] = float(np.mean(values))
            std_lookup[controller_name][metric_key] = float(np.std(values, ddof=0))
        summary_rows.append(summary_row)

    weights = scoring_config["weights"]
    metric_values = {
        metric_key: np.asarray([row[metric_key] for row in summary_rows], dtype=np.float64)
        for metric_key in SCORE_METRICS
    }
    for row in summary_rows:
        component_scores = {}
        weighted_score = 0.0
        for metric_key, _, direction in METRIC_SPECS:
            if metric_key not in SCORE_METRICS:
                continue
            values = metric_values[metric_key]
            minimum = float(values.min())
            maximum = float(values.max())
            if maximum - minimum <= 1e-9:
                score = 100.0
            elif direction == "higher":
                score = 100.0 * (float(row[metric_key]) - minimum) / (maximum - minimum)
            else:
                score = 100.0 * (maximum - float(row[metric_key])) / (maximum - minimum)
            component_scores[metric_key] = float(score)
            weighted_score += float(weights[metric_key]) * float(score)
        row["composite_score"] = float(weighted_score)
        row["radar_scores"] = component_scores

    return summary_rows, std_lookup


def _find_controller_row(summary_rows: list[dict[str, Any]], controller_name: str) -> dict[str, Any] | None:
    for row in summary_rows:
        if row["controller"] == controller_name:
            return row
    return None


def _find_best_row(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(summary_rows, key=lambda row: float(row["composite_score"]))


def _build_baseline_improvements(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    baseline_row = _find_controller_row(summary_rows, BASELINE_CONTROLLER)
    proposed_row = _find_controller_row(summary_rows, PROPOSED_CONTROLLER)
    if baseline_row is None or proposed_row is None:
        return {}

    improvements: dict[str, float] = {}
    for metric_key, _, direction in METRIC_SPECS:
        baseline_value = float(baseline_row[metric_key])
        proposed_value = float(proposed_row[metric_key])
        if abs(baseline_value) <= 1e-9:
            improvements[metric_key] = 0.0
            continue
        if direction == "higher":
            change = (proposed_value - baseline_value) / baseline_value * 100.0
        else:
            change = (baseline_value - proposed_value) / baseline_value * 100.0
        improvements[metric_key] = float(change)
    return improvements


def _write_summary_csv(summary_rows: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for row in summary_rows:
        rows.append(
            {
                "控制算法": row["controller_label"],
                "稳态误差 |e_ss| (μm)": f"{row['steady_state_error_um']:.4f}",
                "调节时间 t_s (s)": f"{row['settling_time_s']:.4f}",
                "涂层均匀度 U_c (%)": f"{row['uniformity_percent']:.2f}",
                "感知误差能量 E_p": f"{row['perception_error_energy']:.4f}",
                "控制平滑度 S_u": f"{row['control_smoothness']:.4f}",
                "超调量 M_p (%)": f"{row['overshoot_percent']:.2f}",
                "综合得分": f"{row['composite_score']:.2f}",
            }
        )
    save_csv(rows, path)


def _write_summary_markdown(summary_rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "控制算法",
        "稳态误差 |e_ss| (μm)↓",
        "调节时间 t_s (s)↓",
        "涂层均匀度 U_c (%)↑",
        "感知误差能量 E_p↓",
        "控制平滑度 S_u↓",
        "超调量 M_p (%)↓",
        "综合得分↑",
    ]
    best_values = {}
    for metric_key, _, direction in METRIC_SPECS:
        values = [float(row[metric_key]) for row in summary_rows]
        best_values[metric_key] = max(values) if direction == "higher" else min(values)
    best_values["composite_score"] = max(float(row["composite_score"]) for row in summary_rows)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        values = [row["controller_label"]]
        for metric_key, _, _ in METRIC_SPECS:
            value = float(row[metric_key])
            if metric_key in {"uniformity_percent", "overshoot_percent"}:
                text = f"{value:.2f}"
            else:
                text = f"{value:.4f}"
            if np.isclose(value, best_values[metric_key]):
                text = f"**{text}**"
            values.append(text)
        score_text = f"{float(row['composite_score']):.2f}"
        if np.isclose(float(row["composite_score"]), best_values["composite_score"]):
            score_text = f"**{score_text}**"
        values.append(score_text)
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("注：表中结果为基于第三章 LUAE 测试集残差序列构造的 10 次闭环仿真实验平均值。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_metric_panels(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for axis, (metric_key, metric_label, direction) in zip(axes, METRIC_SPECS, strict=True):
        values = [float(row[metric_key]) for row in summary_rows]
        labels = [row["controller_label"] for row in summary_rows]
        colors = [CONTROLLER_COLORS[row["controller"]] for row in summary_rows]
        bars = axis.bar(labels, values, color=colors, edgecolor="#2B2B2B", linewidth=0.6)
        axis.set_title(metric_label)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=18)
        note = "越高越优" if direction == "higher" else "越低越优"
        axis.text(0.98, 0.95, note, transform=axis.transAxes, ha="right", va="top", fontsize=9, color="#333333")

        upper = max(values) if values else 1.0
        axis.set_ylim(0.0, upper * 1.18 + 1e-6)
        for bar, value in zip(bars, values, strict=True):
            text = f"{value:.2f}" if metric_key in {"uniformity_percent", "overshoot_percent"} else f"{value:.4f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + upper * 0.03,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for tick_label, row in zip(axis.get_xticklabels(), summary_rows, strict=True):
            if row["controller"] == "A-GC":
                tick_label.set_fontweight("bold")
                tick_label.set_color("#274C77")

    plt.suptitle("五种控制算法关键性能指标对比", y=0.98, fontsize=15)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _plot_radar(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    categories = ["稳态精度", "调节速度", "均匀度", "误差抑制", "平滑性"]
    metric_keys = SCORE_METRICS
    angles = np.linspace(0.0, 2.0 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    figure = plt.figure(figsize=(8.5, 8.5))
    axis = plt.subplot(111, polar=True)
    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(categories)
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_yticklabels(["20", "40", "60", "80", "100"])
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.22)

    for row in summary_rows:
        radar_scores = [float(row["radar_scores"][metric_key]) for metric_key in metric_keys]
        radar_scores.append(radar_scores[0])
        axis.plot(
            angles,
            radar_scores,
            color=CONTROLLER_COLORS[row["controller"]],
            linewidth=2.8 if row["controller"] == "A-GC" else 2.0,
            label=row["controller_label"],
        )
        axis.fill(
            angles,
            radar_scores,
            color=CONTROLLER_COLORS[row["controller"]],
            alpha=0.18 if row["controller"] == "A-GC" else 0.10,
        )

    axis.set_title("五种控制算法综合性能雷达图", pad=24, fontsize=15)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()


def _build_dynamic_response_rows(cycle_rows: list[dict[str, Any]], cycle_time_s: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cycle_rows:
        grouped[(str(row["controller"]), int(row["cycle"]))].append(row)

    response_rows: list[dict[str, Any]] = []
    for controller_name in CONTROLLER_ORDER:
        controller_cycles = sorted(
            [key for key in grouped.keys() if key[0] == controller_name],
            key=lambda item: item[1],
        )
        for _, cycle in controller_cycles:
            rows = grouped[(controller_name, cycle)]
            thickness_values = np.asarray([float(row["thickness_um"]) for row in rows], dtype=np.float64)
            error_values = np.asarray([float(row["thickness_error_um"]) for row in rows], dtype=np.float64)
            target_values = np.asarray(
                [float(row.get("target_thickness_um", 12.0)) for row in rows],
                dtype=np.float64,
            )
            response_rows.append(
                {
                    "controller": controller_name,
                    "controller_label": CONTROLLER_LABELS[controller_name],
                    "cycle": cycle,
                    "time_s": (cycle - 1) * cycle_time_s,
                    "target_thickness_um": float(np.mean(target_values)),
                    "thickness_mean_um": float(np.mean(thickness_values)),
                    "thickness_std_um": float(np.std(thickness_values, ddof=0)),
                    "thickness_min_um": float(np.min(thickness_values)),
                    "thickness_max_um": float(np.max(thickness_values)),
                    "thickness_error_mean_um": float(np.mean(error_values)),
                }
            )
    return response_rows


def _plot_dynamic_response(response_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    plt.figure(figsize=(11.5, 6.8))

    target_drawn = False
    for controller_name in CONTROLLER_ORDER:
        rows = [row for row in response_rows if row["controller"] == controller_name]
        if not rows:
            continue
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
        mean_values = np.asarray([float(row["thickness_mean_um"]) for row in rows], dtype=np.float64)
        std_values = np.asarray([float(row["thickness_std_um"]) for row in rows], dtype=np.float64)
        target_values = np.asarray([float(row["target_thickness_um"]) for row in rows], dtype=np.float64)

        if not target_drawn:
            plt.plot(
                times,
                target_values,
                color="#222222",
                linestyle="--",
                linewidth=1.8,
                label="目标厚度",
            )
            target_drawn = True

        plt.plot(
            times,
            mean_values,
            color=CONTROLLER_COLORS[controller_name],
            linewidth=2.8 if controller_name == "A-GC" else 2.1,
            label=CONTROLLER_LABELS[controller_name],
            zorder=3 if controller_name == "A-GC" else 2,
        )
        plt.fill_between(
            times,
            mean_values - std_values,
            mean_values + std_values,
            color=CONTROLLER_COLORS[controller_name],
            alpha=0.10 if controller_name == "A-GC" else 0.08,
        )

    plt.xlabel("时间 (s)")
    plt.ylabel("涂层厚度 (μm)")
    plt.title("五种控制算法涂层厚度动态响应曲线")
    plt.grid(alpha=0.22)
    plt.legend(ncol=3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _write_experiment_note(
    path: Path,
    *,
    source_csv: Path,
    summary_rows: list[dict[str, Any]],
    phase_specs: list[PhaseSpec],
) -> None:
    best_row = summary_rows[0]
    lines = [
        "# 第四章控制实验说明",
        "",
        "1. 数据来源：直接读取第三章 LUAE 的 `test_predictions.csv`，以重建残差均值为主误差，并融合超阈值异常分数与缺陷面积比构造第四章视觉反馈误差输入。",
        f"2. 信号来源文件：`{source_csv}`。",
        "3. 扰动序列构造：基于 LUAE 输出的 `normal / slight / medium / severe` 四类残差样本，按“基线稳定 - 轻扰动 - 恢复 - 中扰动 - 恢复 - 重扰动 - 恢复”七段工况重采样生成闭环仿真序列。",
        "4. 对比算法：PID、Fuzzy-PID、MA-GC、RL-GC、A-GC（本文算法）。",
        "5. 评价指标：稳态误差、调节时间、涂层均匀度、感知误差能量、控制平滑度、超调量，并进一步计算综合得分。",
        "   其中调节时间定义为：系统进入各恢复阶段后，膜厚误差首次回到容差带内所需的平均时间。",
        f"6. 当前最优方法：{best_row['controller_label']}，综合得分 {best_row['composite_score']:.2f}。",
        "7. 按写作要求，本次结果包不导出各控制方法的涂层厚度动态响应曲线，仅保留表格、关键指标图与综合性能雷达图。",
        "",
        "## 扰动阶段设置",
        "",
    ]
    for phase in phase_specs:
        lines.append(f"- {phase.name}：严重度 `{phase.severity}`，长度 {phase.count} 个控制周期，阶段类型 `{phase.kind}`。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_paper_assets(paper_dir: Path, thesis_roots: list[str | Path]) -> None:
    for root in thesis_roots:
        target_root = ensure_dir(root)
        for file_path in paper_dir.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, target_root / file_path.name)


def _write_experiment_note_v2(
    path: Path,
    *,
    source_csv: Path,
    summary_rows: list[dict[str, Any]],
    phase_specs: list[PhaseSpec],
) -> None:
    best_row = _find_best_row(summary_rows)
    baseline_row = _find_controller_row(summary_rows, BASELINE_CONTROLLER)
    improvements = _build_baseline_improvements(summary_rows)
    lines = [
        "# 第四章控制实验说明",
        "",
        "1. 数据来源：直接读取第三章 LUAE 的 `test_predictions.csv`，以重建残差均值为主误差，并融合超阈值异常分数与缺陷面积比构造第四章视觉反馈误差输入。",
        f"2. 信号来源文件：`{source_csv}`。",
        "3. 扰动序列构造：基于 LUAE 输出的 `normal / slight / medium / severe` 四类残差样本，按“基线稳定 - 轻扰动 - 恢复 - 中扰动 - 恢复 - 重扰动 - 恢复”七段工况重采样生成闭环仿真序列。",
        f"4. 对比算法：PID、Fuzzy-PID、MA-GC、RL-GC、A-GC（本文算法），其中基线方法设定为 {CONTROLLER_LABELS[BASELINE_CONTROLLER]}。",
        "5. 评价指标：稳态误差、调节时间、涂层均匀度、感知误差能量、控制平滑度、超调量，并进一步计算综合得分。",
        "   其中调节时间定义为：系统进入各恢复阶段后，膜厚误差首次回到容差带内所需的平均时间。",
        f"6. 当前最优方法：{best_row['controller_label']}，综合得分 {best_row['composite_score']:.2f}。",
        "7. 本次结果已额外导出五种控制算法的涂层厚度动态响应曲线图及对应统计 CSV，可直接用于第四章结果分析。",
        "",
        "## 相对 PID 基线的改进",
        "",
    ]
    if baseline_row is not None and improvements:
        lines.extend(
            [
                f"- 基线方法：{baseline_row['controller_label']}。",
                (
                    f"- 相较于 PID 基线，A-GC 在稳态误差上降低 {improvements['steady_state_error_um']:.2f}%，"
                    f"调节时间缩短 {improvements['settling_time_s']:.2f}%，"
                    f"涂层均匀度提高 {improvements['uniformity_percent']:.2f}%，"
                    f"感知误差能量降低 {improvements['perception_error_energy']:.2f}%。"
                ),
            ]
        )
    lines.extend(
        [
            "",
        "## 扰动阶段设置",
        "",
        ]
    )
    for phase in phase_specs:
        lines.append(f"- {phase.name}：严重度 `{phase.severity}`，长度 {phase.count} 个控制周期，阶段类型 `{phase.kind}`。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_phase_specs(config: dict[str, Any]) -> list[PhaseSpec]:
    return [
        PhaseSpec(
            name=str(item["name"]),
            severity=str(item["severity"]),
            count=int(item["count"]),
            kind=str(item["kind"]),
        )
        for item in config["sequence_plan"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="第四章：基于 LUAE 残差反馈的五种闭环控制算法对比仿真")
    parser.add_argument("--config", default="configs/chapter4_agc.yaml", help="第四章配置文件路径")
    parser.add_argument("--chapter3-predictions", default=None, help="第三章 LUAE `test_predictions.csv` 的显式路径")
    parser.add_argument("--output-root", default=None, help="第四章输出目录")
    parser.add_argument("--trials", type=int, default=None, help="覆盖配置文件中的仿真重复次数")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.trials is not None:
        config["simulation"]["num_trials"] = int(args.trials)

    chapter3_csv = (
        Path(args.chapter3_predictions)
        if args.chapter3_predictions is not None
        else _resolve_existing_path(config["paths"]["chapter3_prediction_candidates"])
    )
    output_root = ensure_dir(args.output_root if args.output_root is not None else config["paths"]["output_root"])
    raw_dir = ensure_dir(output_root / "raw")
    paper_dir = ensure_dir(output_root / "paper_assets")

    phase_specs = _prepare_phase_specs(config)
    library, library_summary = _load_visual_signal_library(chapter3_csv, config)

    cycle_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for controller_index, controller_name in enumerate(CONTROLLER_ORDER):
        controller_config = config["controllers"][controller_name]
        for trial_index in range(1, int(config["simulation"]["num_trials"]) + 1):
            seed = int(config["seed"]) + controller_index * 1000 + trial_index * 97
            sequence, phase_ranges = _build_trial_sequence(library, phase_specs, seed)
            logs, metrics = _run_single_trial(
                controller_name,
                controller_config,
                config["simulation"],
                config["plant"],
                sequence,
                phase_ranges,
                trial_index,
                seed,
            )
            cycle_rows.extend(logs)
            trial_rows.append(
                {
                    "controller": controller_name,
                    "controller_label": CONTROLLER_LABELS[controller_name],
                    "trial": trial_index,
                    **{metric_key: float(metric_value) for metric_key, metric_value in metrics.items()},
                }
            )

    summary_rows, std_lookup = _aggregate_summary(trial_rows, config["scoring"])
    best_row = _find_best_row(summary_rows)
    baseline_improvements = _build_baseline_improvements(summary_rows)
    response_rows = _build_dynamic_response_rows(cycle_rows, float(config["simulation"]["cycle_time_s"]))
    _write_summary_csv(summary_rows, paper_dir / "Table4-1_五种控制算法性能对比表.csv")
    _write_summary_markdown(summary_rows, paper_dir / "Table4-1_五种控制算法性能对比表_zh.md")
    _plot_metric_panels(summary_rows, paper_dir / "Fig4-1_五种控制算法关键性能指标对比图_zh.png")
    _plot_radar(summary_rows, paper_dir / "Fig4-2_五种控制算法综合性能雷达图_zh.png")
    save_csv(response_rows, paper_dir / "Fig4-3_五种控制算法涂层厚度动态响应曲线.csv")
    _plot_dynamic_response(response_rows, paper_dir / "Fig4-3_五种控制算法涂层厚度动态响应曲线_zh.png")
    _write_experiment_note_v2(
        paper_dir / "附_第四章控制实验说明_zh.md",
        source_csv=chapter3_csv,
        summary_rows=summary_rows,
        phase_specs=phase_specs,
    )

    save_csv(cycle_rows, raw_dir / "chapter4_control_cycle_log.csv")
    save_csv(trial_rows, raw_dir / "chapter4_control_trial_metrics.csv")
    save_json(
        {
            "source_summary": library_summary,
            "phase_specs": [phase.__dict__ for phase in phase_specs],
            "baseline_controller": BASELINE_CONTROLLER,
            "best_controller": best_row["controller"],
            "best_controller_label": best_row["controller_label"],
            "baseline_improvements": baseline_improvements,
            "summary_rows": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "radar_scores"
                }
                for row in summary_rows
            ],
            "std_lookup": std_lookup,
        },
        raw_dir / "chapter4_control_summary.json",
    )

    thesis_roots = [Path(path) for path in config["paths"].get("thesis_output_roots", [])]
    if thesis_roots:
        _copy_paper_assets(paper_dir, thesis_roots)

    print("\n第四章闭环控制仿真完成。")
    print(f"LUAE 残差输入: {chapter3_csv}")
    print(f"原始日志目录: {raw_dir}")
    print(f"论文图表目录: {paper_dir}")
    if thesis_roots:
        print("同步输出目录:")
        for root in thesis_roots:
            print(f"- {root}")
    print(f"最优方法: {best_row['controller_label']} | 综合得分={best_row['composite_score']:.2f}")
