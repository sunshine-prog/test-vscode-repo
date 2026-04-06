from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_WINDOW_CACHE: dict[tuple[int, float, int, str, str], torch.Tensor] = {}


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cache_key = (window_size, sigma, channels, str(device), str(dtype))
    cached = _WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    _WINDOW_CACHE[cache_key] = window
    return window


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    full: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    channels = prediction.size(1)
    window = _gaussian_window(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_pred_target

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    ssim_map = numerator / (denominator + 1e-8)
    score = ssim_map.mean(dim=(1, 2, 3))

    if full:
        return score, ssim_map
    return score


class MSESSIMLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7, ssim_weight: float = 0.3) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        mse_loss = F.mse_loss(prediction, target)
        ssim_score = structural_similarity(prediction, target).mean()
        loss = self.mse_weight * mse_loss + self.ssim_weight * (1.0 - ssim_score)
        return loss, {
            "mse": float(mse_loss.detach().cpu().item()),
            "ssim": float(ssim_score.detach().cpu().item()),
        }


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))
