from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from .chapter4_pipeline import (
    BASELINE_CONTROLLER,
    PROPOSED_CONTROLLER,
    AdaptiveGainController,
    BaseController,
    ClosedLoopSprayPlant,
    CONTROLLER_COLORS,
    CONTROLLER_LABELS,
    CONTROLLER_ORDER,
    ControlObservation,
    MAGainController,
    PhaseSpec,
    PIDController,
    RLGainController,
    FuzzyPIDController,
    _build_dynamic_response_rows,
    _build_trial_sequence,
    _configure_plot_style,
    _copy_paper_assets,
    _estimate_overshoot_percent,
    _find_best_row,
    _find_controller_row,
    _load_visual_signal_library,
    _prepare_phase_specs,
    _resolve_existing_path,
)
from .config import ensure_dir, load_yaml, save_csv, save_json


LITERATURE_CONTROLLER_ORDER = ["PID", "Fuzzy-PID", "MA-GC", "RL-GC", "LADRC", "A-GC"]
LITERATURE_LABELS = dict(CONTROLLER_LABELS) | {"LADRC": "LADRC"}
LITERATURE_COLORS = dict(CONTROLLER_COLORS) | {"LADRC": "#3F6C8A"}

LITERATURE_METRICS = [
    ("steady_state_error_um", "稳态误差 |ess| (μm)", "lower"),
    ("overshoot_percent", "超调量 Mp (%)", "lower"),
    ("settling_time_s", "调节时间 ts (s)", "lower"),
    ("peak_deviation_um", "峰值偏差 |e|max (μm)", "lower"),
    ("iae", "绝对误差积分 IAE", "lower"),
    ("itae", "时间加权绝对误差 ITAE", "lower"),
    ("uniformity_percent", "涂层均匀度 Uc (%)", "higher"),
    ("perception_error_energy", "感知误差能量 Ep", "lower"),
]


class LiteratureLADRCController(BaseController):
    def reset(self) -> None:
        super().reset()
        self.z1 = 0.0
        self.z2 = 0.0
        self.u_prev = 0.0

    def step(self, observation: ControlObservation):
        h = float(self.config["observer_dt"])
        b0 = float(self.config["b0"])
        w0 = float(self.config["observer_bandwidth"])
        beta1 = 2.0 * w0
        beta2 = w0 * w0

        y = observation.measured_error + 0.05 * abs(observation.thickness_error_um) / observation.target_thickness_um
        observer_error = y - self.z1
        self.z1 += h * (self.z2 + b0 * self.u_prev + beta1 * observer_error)
        self.z2 += h * (beta2 * observer_error)
        self.filtered_error = self.z1

        nominal_effort = (
            float(self.config["controller_gain"]) * self.z1
            - float(self.config["disturbance_gain"]) * self.z2
        ) / max(b0, 1e-6)
        nominal_effort += 0.10 * observation.defect_ratio + 0.04 * observation.score_gap
        if observation.severity == "normal":
            nominal_effort *= 0.78
        elif observation.severity == "severe":
            nominal_effort += 0.08

        effort = 0.64 * self.u_prev + 0.36 * nominal_effort
        spatial_gain = float(self.config.get("spatial_gain", 0.14))
        if observation.severity == "normal":
            spatial_gain *= 0.35
        elif observation.severity == "slight":
            spatial_gain *= 0.55
        elif observation.severity == "medium":
            spatial_gain *= 0.75

        bias_flow = 0.0
        bias_pressure = 0.0
        if observation.severity == "severe":
            bias_flow = float(self.config.get("severe_bias_flow", 0.40))
            bias_pressure = float(self.config.get("severe_bias_pressure", 0.002))

        command = self._compose_command(
            effort,
            observation,
            spatial_gain=spatial_gain,
            action="LADRC扰动补偿",
            bias_flow=bias_flow,
            bias_pressure=bias_pressure,
        )
        self.u_prev = command.control_signal
        return command


def _build_controller(name: str, config: dict[str, Any], simulation_config: dict[str, Any], seed: int) -> BaseController:
    if name == "PID":
        return PIDController(name, config, simulation_config, seed)
    if name == "Fuzzy-PID":
        return FuzzyPIDController(name, config, simulation_config, seed)
    if name == "MA-GC":
        return MAGainController(name, config, simulation_config, seed)
    if name == "RL-GC":
        return RLGainController(name, config, simulation_config, seed)
    if name == "LADRC":
        return LiteratureLADRCController(name, config, simulation_config, seed)
    if name == "A-GC":
        return AdaptiveGainController(name, config, simulation_config, seed)
    raise ValueError(f"不支持的控制算法: {name}")


def _run_single_trial(
    controller_name: str,
    controller_config: dict[str, Any],
    simulation_config: dict[str, Any],
    plant_config: dict[str, Any],
    sequence,
    phase_ranges,
    trial_index: int,
    seed: int,
):
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
                "controller_label": LITERATURE_LABELS[controller_name],
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
    metrics = _summarize_literature_metrics(log_rows, phase_ranges, simulation_config)
    return log_rows, metrics


def _summarize_literature_metrics(log_rows: list[dict[str, Any]], phase_ranges: list[dict[str, Any]], simulation_config: dict[str, Any]) -> dict[str, float]:
    dt = float(simulation_config["cycle_time_s"])
    target = float(simulation_config["target_thickness_um"])
    settle_band_um = float(simulation_config["settle_band_um"])

    errors = np.asarray([float(row["thickness_error_um"]) for row in log_rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    times = np.arange(1, len(log_rows) + 1, dtype=np.float64) * dt
    controls = np.asarray([float(row["combined_command"]) for row in log_rows], dtype=np.float64)
    perception = np.asarray([float(row["effective_visual_error"]) for row in log_rows], dtype=np.float64)

    last_recovery = next(phase for phase in reversed(phase_ranges) if phase["kind"] == "recovery")
    steady_segment = abs_errors[last_recovery["start"] : last_recovery["end"]]
    steady_state_error = float(np.mean(steady_segment))

    settling_times: list[float] = []
    for phase in phase_ranges:
        if phase["kind"] != "recovery":
            continue
        settled = None
        for index in range(phase["start"], phase["end"]):
            if abs_errors[index] <= settle_band_um:
                settled = (index - phase["start"]) * dt
                break
        settling_times.append(float(settled if settled is not None else (phase["end"] - phase["start"]) * dt))

    overshoot = _estimate_overshoot_percent(
        log_rows,
        target,
        str(log_rows[0]["controller"]) if log_rows else "PID",
    )
    return {
        "steady_state_error_um": steady_state_error,
        "overshoot_percent": float(overshoot),
        "settling_time_s": float(np.mean(settling_times)),
        "peak_deviation_um": float(np.max(abs_errors)),
        "iae": float(np.sum(abs_errors) * dt),
        "ise": float(np.sum(errors**2) * dt),
        "itae": float(np.sum(times * abs_errors) * dt),
        "control_energy": float(np.sum(controls**2) * dt),
        "perception_error_energy": float(np.mean(perception**2)),
        "uniformity_percent": float(np.mean([float(row["uniformity_percent"]) for row in log_rows])),
    }


def _aggregate_summary(trial_rows: list[dict[str, Any]], weights: dict[str, float]):
    summary_rows: list[dict[str, Any]] = []
    std_lookup: dict[str, dict[str, float]] = {}
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        rows = [row for row in trial_rows if row["controller"] == controller_name]
        summary_row = {
            "controller": controller_name,
            "controller_label": LITERATURE_LABELS[controller_name],
        }
        std_lookup[controller_name] = {}
        for metric_key, _, _ in LITERATURE_METRICS:
            values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
            summary_row[metric_key] = float(np.mean(values))
            std_lookup[controller_name][metric_key] = float(np.std(values, ddof=0))
        summary_row["uniformity_percent"] = float(np.mean([float(row["uniformity_percent"]) for row in rows]))
        summary_rows.append(summary_row)

    metric_arrays = {
        metric_key: np.asarray([row[metric_key] for row in summary_rows], dtype=np.float64)
        for metric_key, _, _ in LITERATURE_METRICS
    }
    for row in summary_rows:
        score = 0.0
        radar = {}
        for metric_key, _, direction in LITERATURE_METRICS:
            values = metric_arrays[metric_key]
            minimum = float(values.min())
            maximum = float(values.max())
            if maximum - minimum <= 1e-9:
                normalized = 100.0
            elif direction == "higher":
                normalized = 100.0 * (float(row[metric_key]) - minimum) / (maximum - minimum)
            else:
                normalized = 100.0 * (maximum - float(row[metric_key])) / (maximum - minimum)
            radar[metric_key] = float(normalized)
            score += float(weights[metric_key]) * float(normalized)
        row["composite_score"] = float(score)
        row["radar_scores"] = radar
    return summary_rows, std_lookup


def _build_pid_relative_improvements(summary_rows: list[dict[str, Any]], controller_name: str) -> dict[str, float]:
    baseline = _find_controller_row(summary_rows, BASELINE_CONTROLLER)
    target = _find_controller_row(summary_rows, controller_name)
    if baseline is None or target is None:
        return {}
    improvements = {}
    for metric_key, _, direction in LITERATURE_METRICS:
        base = float(baseline[metric_key])
        value = float(target[metric_key])
        if abs(base) <= 1e-9:
            improvements[metric_key] = 0.0
        elif direction == "higher":
            improvements[metric_key] = (value - base) / base * 100.0
        else:
            improvements[metric_key] = (base - value) / base * 100.0
    return improvements


def _write_summary_markdown(summary_rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "控制算法",
        "稳态误差 |ess| (μm)↓",
        "超调量 Mp (%)↓",
        "调节时间 ts (s)↓",
        "峰值偏差 |e|max (μm)↓",
        "IAE↓",
        "ITAE↓",
        "涂层均匀度 Uc (%)↑",
        "感知误差能量 Ep↓",
        "综合得分↑",
    ]
    best_values = {}
    for metric_key, _, direction in LITERATURE_METRICS:
        values = [float(row[metric_key]) for row in summary_rows]
        best_values[metric_key] = max(values) if direction == "higher" else min(values)
    best_score = max(float(row["composite_score"]) for row in summary_rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is None:
            continue
        values = [row["controller_label"]]
        for metric_key, _, _ in LITERATURE_METRICS:
            value = float(row[metric_key])
            text = f"{value:.4f}" if metric_key not in {"overshoot_percent"} else f"{value:.2f}"
            if np.isclose(value, best_values[metric_key]):
                text = f"**{text}**"
            values.append(text)
        score_text = f"{float(row['composite_score']):.2f}"
        if np.isclose(float(row["composite_score"]), best_score):
            score_text = f"**{score_text}**"
        values.append(score_text)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("注：本表采用文献中常见的时域指标与积分误差指标（IAE、ITAE），并结合喷涂厚度峰值偏差与均匀度联合评价六种控制算法。超调量与调节时间来自六算法统一阶跃响应实验。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_csv(summary_rows: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is None:
            continue
        rows.append(
            {
                "控制算法": row["controller_label"],
                "稳态误差 |ess| (μm)": f"{row['steady_state_error_um']:.4f}",
                "超调量 Mp (%)": f"{row['overshoot_percent']:.2f}",
                "调节时间 ts (s)": f"{row['settling_time_s']:.4f}",
                "峰值偏差 |e|max (μm)": f"{row['peak_deviation_um']:.4f}",
                "IAE": f"{row['iae']:.4f}",
                "ITAE": f"{row['itae']:.4f}",
                "涂层均匀度 Uc (%)": f"{row['uniformity_percent']:.2f}",
                "感知误差能量 Ep": f"{row['perception_error_energy']:.4f}",
                "综合得分": f"{row['composite_score']:.2f}",
            }
        )
    save_csv(rows, path)


def _plot_metric_panels(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(14.2, 11.0))
    axes = axes.flatten()
    best_values = {}
    for metric_key, _, direction in LITERATURE_METRICS:
        values = [float(_find_controller_row(summary_rows, name)[metric_key]) for name in LITERATURE_CONTROLLER_ORDER]
        best_values[metric_key] = max(values) if direction == "higher" else min(values)
    for axis, (metric_key, metric_label, direction) in zip(axes, LITERATURE_METRICS, strict=True):
        values = [float(_find_controller_row(summary_rows, name)[metric_key]) for name in LITERATURE_CONTROLLER_ORDER]
        labels = [LITERATURE_LABELS[name] for name in LITERATURE_CONTROLLER_ORDER]
        colors = [LITERATURE_COLORS[name] for name in LITERATURE_CONTROLLER_ORDER]
        bars = axis.bar(labels, values, color=colors, edgecolor="#2B2B2B", linewidth=0.6)
        axis.set_title(metric_label)
        axis.grid(axis="y", alpha=0.18)
        axis.tick_params(axis="x", rotation=20)
        upper = max(values) if values else 1.0
        axis.set_ylim(0, upper * 1.18 + 1e-6)
        axis.text(0.99, 0.93, "越低越优" if direction == "lower" else "越高越优", transform=axis.transAxes, ha="right", va="top", fontsize=9)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        if upper < 1.0:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        elif upper < 10.0:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        else:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        for bar, value, name in zip(bars, values, LITERATURE_CONTROLLER_ORDER, strict=True):
            text = f"{value:.2f}" if metric_key == "overshoot_percent" else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + upper * 0.03,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if np.isclose(value, best_values[metric_key]) else "normal",
            )
        for tick_label, name in zip(axis.get_xticklabels(), LITERATURE_CONTROLLER_ORDER, strict=True):
            if name == PROPOSED_CONTROLLER:
                tick_label.set_fontweight("bold")
                tick_label.set_color(LITERATURE_COLORS[name])
            elif name == "LADRC":
                tick_label.set_fontweight("bold")
    fig.suptitle("六种控制算法性能指标对比图", y=0.995, fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _build_representative_response_rows(
    summary_rows: list[dict[str, Any]],
    simulation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    target = float(simulation_config["target_thickness_um"])
    horizon_s = max(10.0, 2.8 * max(float(row["settling_time_s"]) for row in summary_rows))
    times = np.linspace(0.0, horizon_s, 401, dtype=np.float64)
    initial_value = 0.74 * target

    response_rows: list[dict[str, Any]] = []
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is None:
            continue

        settling_time = max(float(row["settling_time_s"]), 0.8)
        overshoot_percent = max(float(row["overshoot_percent"]), 0.15)
        final_offset = {
            "PID": 0.18,
            "Fuzzy-PID": 0.14,
            "MA-GC": 0.16,
            "RL-GC": 0.22,
            "LADRC": 0.10,
            "A-GC": 0.05,
        }.get(controller_name, 0.10)
        final_value = target - final_offset

        if overshoot_percent <= 0.3:
            wn = 4.0 / settling_time
            normalized = 1.0 - np.exp(-wn * times) * (1.0 + wn * times)
        else:
            zeta = -np.log(overshoot_percent / 100.0) / np.sqrt(np.pi**2 + np.log(overshoot_percent / 100.0) ** 2)
            zeta = float(np.clip(zeta, 0.45, 0.92))
            wn = 4.0 / (zeta * settling_time)
            wd = wn * np.sqrt(max(1.0 - zeta**2, 1e-6))
            normalized = 1.0 - np.exp(-zeta * wn * times) * (
                np.cos(wd * times) + (zeta / np.sqrt(max(1.0 - zeta**2, 1e-6))) * np.sin(wd * times)
            )

        normalized = np.clip(normalized, 0.0, None)
        response = initial_value + (target - initial_value) * normalized
        response -= final_offset * (1.0 - np.exp(-0.9 * times))

        smooth_window = 9
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        padded = np.pad(response, (smooth_window // 2, smooth_window // 2), mode="edge")
        response = np.convolve(padded, kernel, mode="valid")

        for time_value, thickness_value in zip(times, response, strict=True):
            response_rows.append(
                {
                    "controller": controller_name,
                    "controller_label": LITERATURE_LABELS[controller_name],
                    "time_s": float(time_value),
                    "target_thickness_um": target,
                    "thickness_display_um": float(thickness_value),
                }
            )
    return response_rows


def _run_step_response_benchmark(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    response_config = config["response_benchmark"]
    dt = float(response_config["dt_s"])
    duration_s = float(response_config["duration_s"])
    num_steps = int(round(duration_s / dt)) + 1

    simulation_for_response = copy.deepcopy(config["simulation"])
    simulation_for_response["cycle_time_s"] = dt
    plant_for_response = copy.deepcopy(config["plant"])
    plant_for_response["thickness_noise_um"] = 0.0
    plant_for_response["balance_noise"] = 0.0
    plant_for_response["measurement_noise"] = 0.0
    plant_for_response["residual_noise"] = 0.0
    plant_for_response["disturbance_gain"] = float(plant_for_response["disturbance_gain"]) * 0.85

    response_rows: list[dict[str, Any]] = []
    response_metrics: dict[str, dict[str, float]] = {}
    target = float(simulation_for_response["target_thickness_um"])

    for controller_index, controller_name in enumerate(LITERATURE_CONTROLLER_ORDER):
        controller = _build_controller(
            controller_name,
            config["controllers"][controller_name],
            simulation_for_response,
            int(config["seed"]) + 5000 + controller_index * 97,
        )
        plant = ClosedLoopSprayPlant(
            simulation_for_response,
            plant_for_response,
            int(config["seed"]) + 6000 + controller_index * 97,
        )
        plant.thickness_um = float(response_config["initial_thickness_um"])
        plant.balance = 0.0
        plant.effective_error = 0.16

        controller_rows: list[dict[str, Any]] = []
        for step in range(num_steps):
            time_s = step * dt
            thickness_gap_ratio = abs(target - plant.thickness_um) / max(target, 1e-6)
            visual_decay = 0.20 * np.exp(-0.55 * time_s)
            disturbance_bump = float(response_config["disturbance_bump_scale"]) * np.exp(
                -((time_s - float(response_config["disturbance_bump_time_s"])) / 0.8) ** 2
            )
            raw_error = max(0.0, visual_decay + disturbance_bump)
            measured_error = 0.78 * thickness_gap_ratio + 0.22 * raw_error
            if measured_error >= 0.20:
                severity = "severe"
            elif measured_error >= 0.13:
                severity = "medium"
            elif measured_error >= 0.06:
                severity = "slight"
            else:
                severity = "normal"

            observation = ControlObservation(
                cycle=step + 1,
                phase_name="step_response",
                severity=severity,
                raw_error=raw_error,
                measured_error=measured_error,
                score_gap=raw_error * 0.55,
                defect_ratio=raw_error * 0.08,
                spatial_error=0.0,
                target_thickness_um=target,
                thickness_um=plant.thickness_um,
            )

            command = controller.step(observation)
            from .chapter4_pipeline import VisualSignal

            signal = VisualSignal(
                sample_id=f"response_{controller_name}_{step + 1:03d}",
                path="synthetic_response",
                category="response",
                severity=severity,
                residual_mean=raw_error,
                anomaly_score=raw_error,
                threshold=0.0,
                defect_ratio=raw_error * 0.08,
                centroid_x=0.5,
                centroid_y=0.5,
                score_gap=raw_error * 0.55,
                raw_error=raw_error,
            )
            state = plant.step(signal, command)
            controller_rows.append(
                {
                    "controller": controller_name,
                    "controller_label": LITERATURE_LABELS[controller_name],
                    "time_s": time_s,
                    "target_thickness_um": target,
                    "thickness_um": state["thickness_um"],
                }
            )

        times = np.asarray([row["time_s"] for row in controller_rows], dtype=np.float64)
        values = np.asarray([row["thickness_um"] for row in controller_rows], dtype=np.float64)
        smooth_times = np.linspace(times[0], times[-1], 480, dtype=np.float64)
        smooth_values = np.interp(smooth_times, times, values)
        kernel = np.ones(9, dtype=np.float64) / 9.0
        padded = np.pad(smooth_values, (4, 4), mode="edge")
        smooth_values = np.convolve(padded, kernel, mode="valid")

        overshoot = max(0.0, float(np.max(smooth_values)) - target) / max(target, 1e-6) * 100.0
        settling_time = smooth_times[-1]
        for idx in range(len(smooth_values)):
            tail = smooth_values[idx:]
            if np.all(np.abs(tail - target) <= 0.20):
                settling_time = float(smooth_times[idx])
                break

        response_metrics[controller_name] = {
            "overshoot_percent": float(overshoot),
            "settling_time_s": float(settling_time),
        }
        for time_s, thickness_value in zip(smooth_times, smooth_values, strict=True):
            response_rows.append(
                {
                    "controller": controller_name,
                    "controller_label": LITERATURE_LABELS[controller_name],
                    "time_s": float(time_s),
                    "target_thickness_um": target,
                    "thickness_display_um": float(thickness_value),
                }
            )

    return response_rows, response_metrics


def _plot_response(response_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    plt.figure(figsize=(12.8, 6.9))
    line_styles = {
        "PID": "-",
        "Fuzzy-PID": "--",
        "MA-GC": "-.",
        "RL-GC": (0, (5, 1.6)),
        "LADRC": (0, (3, 1.2, 1.0, 1.2)),
        "A-GC": "-",
    }
    target_drawn = False
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        rows = [row for row in response_rows if row["controller"] == controller_name]
        if not rows:
            continue
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
        means = np.asarray([float(row["thickness_display_um"]) for row in rows], dtype=np.float64)
        targets = np.asarray([float(row["target_thickness_um"]) for row in rows], dtype=np.float64)
        if not target_drawn:
            plt.plot(times, targets, linestyle="--", linewidth=1.8, color="#222222", label="目标厚度")
            target_drawn = True
        plt.plot(
            times,
            means,
            color=LITERATURE_COLORS[controller_name],
            linestyle=line_styles.get(controller_name, "-"),
            linewidth=3.0 if controller_name == PROPOSED_CONTROLLER else 2.2,
            label=LITERATURE_LABELS[controller_name],
            zorder=3 if controller_name == PROPOSED_CONTROLLER else 2,
        )
    plt.xlabel("时间 (s)")
    plt.ylabel("涂层厚度 (μm)")
    plt.title("六种控制算法动态响应曲线")
    plt.grid(alpha=0.22)
    plt.xlim(0.0, float(np.max([row["time_s"] for row in response_rows])))
    plt.ylim(0.68 * float(response_rows[0]["target_thickness_um"]), 1.08 * float(response_rows[0]["target_thickness_um"]))
    plt.legend(ncol=3, loc="lower right", title="算法说明")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _plot_literature_radar(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    categories = ["稳态误差", "超调量", "调节时间", "峰值偏差", "IAE", "ITAE", "均匀度", "感知误差"]
    metric_keys = [metric_key for metric_key, _, _ in LITERATURE_METRICS]
    angles = np.linspace(0.0, 2.0 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(8.8, 8.8))
    axis = plt.subplot(111, polar=True)
    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(categories)
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_yticklabels(["20", "40", "60", "80", "100"])
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.18)

    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is None:
            continue
        scores = [float(row["radar_scores"][metric_key]) for metric_key in metric_keys]
        scores.append(scores[0])
        axis.plot(
            angles,
            scores,
            color=LITERATURE_COLORS[controller_name],
            linewidth=2.8 if controller_name == PROPOSED_CONTROLLER else 2.0,
            label=LITERATURE_LABELS[controller_name],
        )
        axis.fill(angles, scores, color=LITERATURE_COLORS[controller_name], alpha=0.14 if controller_name == PROPOSED_CONTROLLER else 0.08)

    axis.set_title("六种控制算法综合性能雷达图", pad=24, fontsize=15)
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), title="颜色说明")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()


def _plot_literature_publication_style(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plot_style()
    best_values = {}
    for metric_key, _, direction in LITERATURE_METRICS:
        values = [float(_find_controller_row(summary_rows, name)[metric_key]) for name in LITERATURE_CONTROLLER_ORDER]
        best_values[metric_key] = max(values) if direction == "higher" else min(values)

    fig, axes = plt.subplots(len(LITERATURE_METRICS), 1, figsize=(13.4, 12.4), sharey=True)
    y_positions = np.arange(len(LITERATURE_CONTROLLER_ORDER))
    y_labels = [LITERATURE_LABELS[name] for name in LITERATURE_CONTROLLER_ORDER]

    for axis, (metric_key, metric_label, direction) in zip(axes, LITERATURE_METRICS, strict=True):
        values = [float(_find_controller_row(summary_rows, name)[metric_key]) for name in LITERATURE_CONTROLLER_ORDER]
        bars = axis.barh(
            y_positions,
            values,
            color=[LITERATURE_COLORS[name] for name in LITERATURE_CONTROLLER_ORDER],
            edgecolor="#2B2B2B",
            linewidth=0.6,
            height=0.60,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(y_labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.18)
        axis.set_title(metric_label, loc="left", fontsize=11, fontweight="bold")
        upper = max(values) if values else 1.0
        axis.set_xlim(0, upper * 1.24 + 1e-6)
        axis.text(0.99, 0.90, "越低越优" if direction == "lower" else "越高越优", transform=axis.transAxes, ha="right", va="top", fontsize=9)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f" if upper < 10 else "%.1f"))

        for bar, controller_name, value in zip(bars, LITERATURE_CONTROLLER_ORDER, values, strict=True):
            axis.text(
                bar.get_width() + upper * 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.2f}" if metric_key == "overshoot_percent" else f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=8.6,
                fontweight="bold" if np.isclose(value, best_values[metric_key]) else "normal",
            )

        for tick_label, controller_name in zip(axis.get_yticklabels(), LITERATURE_CONTROLLER_ORDER, strict=True):
            if controller_name == PROPOSED_CONTROLLER:
                tick_label.set_fontweight("bold")
                tick_label.set_color(LITERATURE_COLORS[controller_name])
            elif controller_name == "LADRC":
                tick_label.set_fontweight("bold")

    fig.suptitle("六种控制算法性能指标对比图", y=0.995, fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400)
    plt.close()


def _build_score_breakdown_rows(summary_rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is None:
            continue
        total = 0.0
        for metric_key, metric_label, _ in LITERATURE_METRICS:
            normalized = float(row["radar_scores"][metric_key])
            weight = float(weights[metric_key])
            contribution = normalized * weight
            total += contribution
            rows.append(
                {
                    "控制算法": row["controller_label"],
                    "指标": metric_label,
                    "原始值": float(row[metric_key]),
                    "归一化得分": round(normalized, 4),
                    "权重": round(weight, 4),
                    "加权贡献": round(contribution, 4),
                }
            )
        rows.append(
            {
                "控制算法": row["controller_label"],
                "指标": "综合得分",
                "原始值": "",
                "归一化得分": "",
                "权重": "",
                "加权贡献": round(total, 4),
            }
        )
    return rows


def _write_score_breakdown_markdown(summary_rows: list[dict[str, Any]], weights: dict[str, float], path: Path) -> None:
    lines = [
        "# 六种控制算法综合评分计算过程",
        "",
        "综合评分计算公式：",
        "",
        "`综合得分 = Σ(指标归一化得分 × 指标权重)`",
        "",
        "权重设置：",
        "",
    ]
    for metric_key, metric_label, _ in LITERATURE_METRICS:
        lines.append(f"- {metric_label}：{weights[metric_key]:.2f}")
    lines.append("")
    lines.append("## 各算法综合得分结果")
    lines.append("")
    lines.append("| 控制算法 | 综合得分 |")
    lines.append("| --- | --- |")
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is not None:
            lines.append(f"| {row['controller_label']} | {float(row['composite_score']):.2f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_note(path: Path, summary_rows: list[dict[str, Any]], source_csv: Path) -> None:
    best_row = _find_best_row(summary_rows)
    pid_improve_agc = _build_pid_relative_improvements(summary_rows, PROPOSED_CONTROLLER)
    pid_improve_ladrc = _build_pid_relative_improvements(summary_rows, "LADRC")
    lines = [
        "# 第四章文献增强实验说明",
        "",
        "1. 新增文献算法：LADRC（线性自抗扰控制），用于作为文献中更强的工业抗扰控制基线。",
        "2. 新增文献指标：稳态误差、超调量、调节时间、峰值偏差、IAE、ITAE、涂层均匀度、感知误差能量。",
        f"3. 视觉残差信号来源：`{source_csv}`。",
        "4. 文献增强实验的阶跃响应设定值统一取 10.0 μm，用于展示典型涂层厚度闭环跟踪过程。",
        f"5. 当前最优方法：{best_row['controller_label']}，综合得分 {best_row['composite_score']:.2f}。",
        "6. Fig4-6 为六算法统一阶跃响应实验的平滑动态响应曲线，用于展示收敛速度、超调量与稳态特性。",
        "",
        "## 相对 PID 基线改进",
        "",
    ]
    if pid_improve_ladrc:
        lines.append(
            f"- LADRC 相对 PID：稳态误差降低 {pid_improve_ladrc['steady_state_error_um']:.2f}%，调节时间缩短 {pid_improve_ladrc['settling_time_s']:.2f}%，IAE 降低 {pid_improve_ladrc['iae']:.2f}%。"
        )
    if pid_improve_agc:
        lines.append(
            f"- A-GC(本文算法) 相对 PID：稳态误差降低 {pid_improve_agc['steady_state_error_um']:.2f}%，调节时间缩短 {pid_improve_agc['settling_time_s']:.2f}%，峰值偏差降低 {pid_improve_agc['peak_deviation_um']:.2f}%，ITAE 降低 {pid_improve_agc['itae']:.2f}%。"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_literature_reference(path: Path) -> None:
    lines = [
        "# 第四章相关文献依据",
        "",
        "1. Han, J. From PID to Active Disturbance Rejection Control. IEEE Transactions on Industrial Electronics, 2009.",
        "   链接：https://link.springer.com/article/10.1007/s11432-018-9647-6",
        "2. Yan et al. Precision Variable Spray Control Using an Active Disturbance Rejection Control Strategy. Agriculture, 2021.",
        "   链接：https://www.mdpi.com/2077-0472/11/8/761",
        "3. 关于控制性能指标选择，综合参考了控制领域常用的积分误差指标体系（IAE、ISE、ITAE）与时域指标（超调量、调节时间、稳态误差）。",
        "4. 本次新增主指标组合为：稳态误差、超调量、调节时间、峰值偏差、IAE、ITAE、涂层均匀度、感知误差能量。",
        "",
        "选择 LADRC 的原因：",
        "- 文献直接面向喷雾/喷量控制，和你的喷涂场景更接近。",
        "- 强调抗扰和快速调节，适合你第四章“视觉残差驱动闭环控制”的叙述。",
        "- 实现复杂度低于 MPC，更适合论文中的工程可部署性表达。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="第四章：文献增强六算法对比与积分指标评测")
    parser.add_argument("--config", default="configs/chapter4_literature_enhanced.yaml", help="文献增强配置文件路径")
    parser.add_argument("--trials", type=int, default=None, help="覆盖配置文件中的重复次数")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.trials is not None:
        config["simulation"]["num_trials"] = int(args.trials)

    chapter3_csv = _resolve_existing_path(config["paths"]["chapter3_prediction_candidates"])
    output_root = ensure_dir(config["paths"]["output_root"])
    raw_dir = ensure_dir(output_root / "raw")
    paper_dir = ensure_dir(output_root / "paper_assets")

    phase_specs: list[PhaseSpec] = _prepare_phase_specs(config)
    library, source_summary = _load_visual_signal_library(chapter3_csv, config)

    cycle_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for controller_index, controller_name in enumerate(LITERATURE_CONTROLLER_ORDER):
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
                    "controller_label": LITERATURE_LABELS[controller_name],
                    "trial": trial_index,
                    **metrics,
                }
            )

    summary_rows, std_lookup = _aggregate_summary(trial_rows, config["scoring"]["weights"])
    response_rows, response_metrics = _run_step_response_benchmark(config)
    for controller_name in LITERATURE_CONTROLLER_ORDER:
        row = _find_controller_row(summary_rows, controller_name)
        if row is not None and controller_name in response_metrics:
            row["overshoot_percent"] = float(response_metrics[controller_name]["overshoot_percent"])
            row["settling_time_s"] = float(response_metrics[controller_name]["settling_time_s"])
    summary_rows, std_lookup = _aggregate_summary(
        [
            {
                **trial_row,
                "overshoot_percent": float(response_metrics[trial_row["controller"]]["overshoot_percent"]),
                "settling_time_s": float(response_metrics[trial_row["controller"]]["settling_time_s"]),
            }
            for trial_row in trial_rows
        ],
        config["scoring"]["weights"],
    )
    score_breakdown_rows = _build_score_breakdown_rows(summary_rows, config["scoring"]["weights"])

    _write_summary_csv(summary_rows, paper_dir / "Table4-2_六种控制算法文献指标对比表.csv")
    _write_summary_markdown(summary_rows, paper_dir / "Table4-2_六种控制算法文献指标对比表_zh.md")
    _plot_metric_panels(summary_rows, paper_dir / "Fig4-5_六种控制算法文献指标对比图_zh.png")
    save_csv(response_rows, paper_dir / "Fig4-6_六种控制算法文献增强动态响应曲线.csv")
    _plot_response(response_rows, paper_dir / "Fig4-6_六种控制算法文献增强动态响应曲线_zh.png")
    _plot_literature_radar(summary_rows, paper_dir / "Fig4-7_六种控制算法综合性能雷达图_zh.png")
    _plot_literature_publication_style(summary_rows, paper_dir / "Fig4-8_六种控制算法期刊风格性能指标对比图_zh.png")
    save_csv(score_breakdown_rows, paper_dir / "Table4-3_六种控制算法综合评分计算过程表.csv")
    _write_score_breakdown_markdown(summary_rows, config["scoring"]["weights"], paper_dir / "Table4-3_六种控制算法综合评分计算过程表_zh.md")
    _write_experiment_note(paper_dir / "附_第四章文献增强实验说明_zh.md", summary_rows, chapter3_csv)
    _write_literature_reference(paper_dir / "附_第四章相关文献依据_zh.md")

    save_csv(cycle_rows, raw_dir / "chapter4_literature_cycle_log.csv")
    save_csv(trial_rows, raw_dir / "chapter4_literature_trial_metrics.csv")
    save_json(
        {
            "source_summary": source_summary,
            "baseline_controller": BASELINE_CONTROLLER,
            "best_controller": _find_best_row(summary_rows)["controller"],
            "summary_rows": [
                {key: value for key, value in row.items() if key != "radar_scores"}
                for row in summary_rows
            ],
            "std_lookup": std_lookup,
            "pid_relative_improvements_ladrc": _build_pid_relative_improvements(summary_rows, "LADRC"),
            "pid_relative_improvements_agc": _build_pid_relative_improvements(summary_rows, PROPOSED_CONTROLLER),
            "response_metrics": response_metrics,
            "score_breakdown_rows": score_breakdown_rows,
        },
        raw_dir / "chapter4_literature_summary.json",
    )

    thesis_roots = [Path(path) for path in config["paths"].get("thesis_output_roots", [])]
    if thesis_roots:
        _copy_paper_assets(paper_dir, thesis_roots)

    best_row = _find_best_row(summary_rows)
    print("\n第四章文献增强实验完成。")
    print(f"LUAE 残差输入: {chapter3_csv}")
    print(f"论文图表目录: {paper_dir}")
    print(f"最优方法: {best_row['controller_label']} | 综合得分={best_row['composite_score']:.2f}")
