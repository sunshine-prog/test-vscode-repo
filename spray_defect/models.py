from __future__ import annotations

import torch
import torch.nn as nn


def _build_norm(channels: int, norm_type: str, group_count: int) -> nn.Module:
    normalized = norm_type.strip().lower()
    if normalized == "batchnorm":
        return nn.BatchNorm2d(channels)
    if normalized == "groupnorm":
        groups = max(1, min(group_count, channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized == "instancenorm":
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type!r}")


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str, group_count: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            _build_norm(in_channels, norm_type, group_count),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _build_norm(out_channels, norm_type, group_count),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str, group_count: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableBlock(in_channels, out_channels, norm_type, group_count),
            DepthwiseSeparableBlock(out_channels, out_channels, norm_type, group_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_type: str, group_count: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            DepthwiseSeparableBlock(out_channels + skip_channels, out_channels, norm_type, group_count),
            DepthwiseSeparableBlock(out_channels, out_channels, norm_type, group_count),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LightweightUNetAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        norm_type: str = "batchnorm",
        group_count: int = 8,
    ) -> None:
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_channels, norm_type, group_count)
        self.enc2 = EncoderStage(base_channels, base_channels * 2, norm_type, group_count)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4, norm_type, group_count)
        self.bottleneck = EncoderStage(base_channels * 4, base_channels * 8, norm_type, group_count)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dec3 = DecoderStage(base_channels * 8, base_channels * 4, base_channels * 4, norm_type, group_count)
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2, norm_type, group_count)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels, norm_type, group_count)

        self.output = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        bottleneck = self.bottleneck(self.pool(skip3))

        x = self.dec3(bottleneck, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
