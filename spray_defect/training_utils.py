from __future__ import annotations

import copy
from typing import Any

import torch


class ExponentialMovingAverage:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        enabled: bool = False,
        decay: float = 0.995,
    ) -> None:
        self.enabled = bool(enabled)
        self.decay = float(decay)
        self.num_updates = 0
        self.ema_model: torch.nn.Module | None = None

        if not self.enabled:
            return

        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        if not self.enabled or self.ema_model is None:
            return

        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

        with torch.no_grad():
            ema_state = self.ema_model.state_dict()
            model_state = model.state_dict()
            for key, ema_value in ema_state.items():
                model_value = model_state[key].detach()
                if torch.is_floating_point(ema_value):
                    ema_value.lerp_(model_value, 1.0 - decay)
                else:
                    ema_value.copy_(model_value)

    def get_eval_model(self, fallback_model: torch.nn.Module) -> torch.nn.Module:
        if self.enabled and self.ema_model is not None:
            return self.ema_model
        return fallback_model


def apply_denoising_noise(images: torch.Tensor, denoising_config: dict[str, Any] | None) -> torch.Tensor:
    if not denoising_config or not bool(denoising_config.get("enabled", False)):
        return images

    noisy = images.clone()
    gaussian_std = float(denoising_config.get("gaussian_std", 0.0))
    speckle_std = float(denoising_config.get("speckle_std", 0.0))
    cutout_prob = float(denoising_config.get("cutout_prob", 0.0))
    cutout_min_ratio = float(denoising_config.get("cutout_min_ratio", 0.05))
    cutout_max_ratio = float(denoising_config.get("cutout_max_ratio", 0.15))
    cutout_blocks = max(int(denoising_config.get("cutout_blocks", 1)), 1)
    cutout_fill_mode = str(denoising_config.get("cutout_fill_mode", "mean")).strip().lower()

    if gaussian_std > 0.0:
        noisy = noisy + torch.randn_like(noisy) * gaussian_std

    if speckle_std > 0.0:
        noisy = noisy + noisy * torch.randn_like(noisy) * speckle_std

    if cutout_prob > 0.0:
        _, _, height, width = noisy.shape
        for index in range(noisy.size(0)):
            if float(torch.rand((), device=noisy.device).item()) >= cutout_prob:
                continue

            for _ in range(cutout_blocks):
                ratio_h = float(torch.empty((), device=noisy.device).uniform_(cutout_min_ratio, cutout_max_ratio).item())
                ratio_w = float(torch.empty((), device=noisy.device).uniform_(cutout_min_ratio, cutout_max_ratio).item())
                cut_h = max(1, int(round(height * ratio_h)))
                cut_w = max(1, int(round(width * ratio_w)))
                top = int(torch.randint(0, max(height - cut_h + 1, 1), (), device=noisy.device).item())
                left = int(torch.randint(0, max(width - cut_w + 1, 1), (), device=noisy.device).item())

                if cutout_fill_mode == "zero":
                    fill_value = 0.0
                else:
                    fill_value = noisy[index : index + 1].mean()
                noisy[index, :, top : top + cut_h, left : left + cut_w] = fill_value

    return noisy.clamp(0.0, 1.0)
