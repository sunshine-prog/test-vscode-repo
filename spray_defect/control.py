from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .anomaly import DetectionSignal


@dataclass
class ControlCommand:
    flow_g_per_15s: float
    pressure_mpa: float
    spray_angle_deg: float
    action: str
    filtered_error: float
    delta_error: float


class AdaptiveGainController:
    def __init__(self, controller_config: dict[str, Any]) -> None:
        self.config = controller_config
        self.base = controller_config["base_parameters"]
        self.limits = controller_config["limits"]
        self.scales = controller_config["command_scales"]
        self.severity_gain = controller_config["severity_gain"]
        self.severity_bias = controller_config["severity_bias"]
        self.ema_alpha = controller_config.get("ema_alpha", 0.72)
        self.integral_limit = controller_config.get("integral_limit", 3.0)
        self.adaptive_rate = controller_config.get("adaptive_rate", 0.35)
        self.derivative_rate = controller_config.get("derivative_rate", 0.2)

        self.filtered_error = 0.0
        self.previous_error = 0.0
        self.integral_error = 0.0

    def step(self, signal: DetectionSignal) -> ControlCommand:
        raw_error = max(0.0, signal.anomaly_score - signal.threshold)
        self.filtered_error = self.ema_alpha * self.filtered_error + (1.0 - self.ema_alpha) * raw_error
        delta_error = self.filtered_error - self.previous_error
        self.previous_error = self.filtered_error
        self.integral_error = float(
            np.clip(
                self.integral_error + self.filtered_error,
                -self.integral_limit,
                self.integral_limit,
            )
        )

        severity = signal.severity
        gain = self.severity_gain.get(severity, 0.0)
        bias = self.severity_bias.get(severity, {"flow": 0.0, "pressure": 0.0, "angle": 0.0})
        adaptive_factor = 1.0 + self.adaptive_rate * max(delta_error, 0.0) + 1.5 * signal.defect_ratio
        control_core = self.filtered_error + self.derivative_rate * delta_error + 0.15 * self.integral_error
        control_core *= gain * adaptive_factor

        flow = self.base["flow_g_per_15s"] + bias["flow"] + self.scales["flow"] * control_core
        pressure = self.base["pressure_mpa"] + bias["pressure"] + self.scales["pressure"] * control_core
        angle = self.base["spray_angle_deg"] - bias["angle"] - self.scales["angle"] * control_core

        flow = float(np.clip(flow, *self.limits["flow_g_per_15s"]))
        pressure = float(np.clip(pressure, *self.limits["pressure_mpa"]))
        angle = float(np.clip(angle, *self.limits["spray_angle_deg"]))

        if severity == "severe" and signal.defect_ratio > 0.10:
            action = "alarm_and_global_respray"
        elif signal.predicted_label == 1:
            action = "local_respray"
        else:
            action = "hold"

        return ControlCommand(
            flow_g_per_15s=flow,
            pressure_mpa=pressure,
            spray_angle_deg=angle,
            action=action,
            filtered_error=self.filtered_error,
            delta_error=delta_error,
        )


class SprayProcessSimulator:
    def __init__(self, plant_config: dict[str, Any], base_parameters: dict[str, float], seed: int = 42) -> None:
        self.dt = plant_config.get("dt", 0.1)
        self.gain = plant_config.get("gain", 0.85)
        self.time_constant = plant_config.get("time_constant", 0.65)
        self.target_thickness = plant_config.get("target_thickness", 1.0)
        self.disturbance_scale = plant_config.get("disturbance_scale", 0.08)
        self.base = base_parameters
        self.rng = np.random.default_rng(seed)

        self.thickness = self.target_thickness
        self.uniformity = 1.0

    def step(self, command: ControlCommand, signal: DetectionSignal) -> dict[str, float]:
        flow_delta = (command.flow_g_per_15s - self.base["flow_g_per_15s"]) / max(self.base["flow_g_per_15s"], 1e-8)
        pressure_delta = (command.pressure_mpa - self.base["pressure_mpa"]) / max(self.base["pressure_mpa"], 1e-8)
        angle_delta = (self.base["spray_angle_deg"] - command.spray_angle_deg) / max(self.base["spray_angle_deg"], 1e-8)
        disturbance = float(self.rng.normal(0.0, self.disturbance_scale))

        control_effect = 0.55 * flow_delta + 0.30 * pressure_delta + 0.15 * angle_delta
        defect_penalty = max(0.0, signal.anomaly_score - signal.threshold) + 1.2 * signal.defect_ratio
        derivative = (
            -(self.thickness - self.target_thickness) / self.time_constant
            + self.gain * control_effect
            - 0.40 * defect_penalty
            + disturbance
        )
        self.thickness = max(0.0, self.thickness + self.dt * derivative)
        steady_error = self.target_thickness - self.thickness
        self.uniformity = float(np.clip(1.0 - abs(steady_error) - 0.6 * signal.defect_ratio, 0.0, 1.0))

        return {
            "thickness": float(self.thickness),
            "steady_error": float(steady_error),
            "uniformity": self.uniformity,
        }
