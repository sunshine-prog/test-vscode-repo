from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import resolve_dataloader_kwargs


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    category: str
    patch_index: int
    patch_x: int
    patch_y: int
    patch_w: int
    patch_h: int
    base_width: int
    base_height: int
    cache_path: Path | None = None


class SprayImageDatasetV2(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int = 256,
        roi: list[int] | None = None,
        normalize_orientation: bool = True,
        auto_crop: bool = True,
        clahe: bool = True,
        gaussian_blur: bool = True,
        median_blur: bool = True,
        pad_mode: str = "mean",
        patching_enabled: bool = False,
        patch_size: int = 768,
        patch_stride: int = 640,
        max_patches_per_image: int | None = None,
        min_patch_std: float = 8.0,
        cache_enabled: bool = False,
        cache_dir: str | Path | None = None,
        augment_mode: str = "none",
        fft_noise_scale: float = 0.06,
        rotation_deg: float = 6.0,
        brightness_jitter: float = 0.08,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.roi = roi
        self.normalize_orientation = normalize_orientation
        self.auto_crop = auto_crop
        self.clahe = clahe
        self.gaussian_blur = gaussian_blur
        self.median_blur = median_blur
        self.pad_mode = pad_mode
        self.patching_enabled = patching_enabled
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.max_patches_per_image = max_patches_per_image
        self.min_patch_std = min_patch_std
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.augment_mode = augment_mode
        self.fft_noise_scale = fft_noise_scale
        self.rotation_deg = rotation_deg
        self.brightness_jitter = brightness_jitter

        categories = ["normal"] if split in {"train", "val"} else ["normal", "defect"]
        image_paths: list[tuple[Path, int, str]] = []
        for category in categories:
            category_dir = self.root / split / category
            if not category_dir.exists():
                continue
            label = 1 if category == "defect" else 0
            for path in sorted(category_dir.glob("*.jpg")):
                image_paths.append((path, label, category))

        if max_samples is not None:
            image_paths = image_paths[:max_samples]

        self.samples: list[SampleRecord] = []
        for path, label, category in image_paths:
            self.samples.extend(self._build_records(path, label, category))

        if not self.samples:
            raise FileNotFoundError(f"在 {self.root / split} 下没有找到可用图片。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if sample.cache_path is not None and sample.cache_path.exists():
            patch = self._read_image(sample.cache_path)
        else:
            image = self._read_image(sample.path)
            image = self._prepare_base_image(image)
            patch = image[
                sample.patch_y : sample.patch_y + sample.patch_h,
                sample.patch_x : sample.patch_x + sample.patch_w,
            ]
            patch = self._finalize_patch(patch)
        if self.split == "train":
            patch = self._augment(patch)

        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "image": tensor,
            "label": sample.label,
            "path": str(sample.path),
            "sample_id": sample.path.stem,
            "category": sample.category,
            "patch_index": sample.patch_index,
            "patch_x": sample.patch_x,
            "patch_y": sample.patch_y,
            "patch_w": sample.patch_w,
            "patch_h": sample.patch_h,
            "base_width": sample.base_width,
            "base_height": sample.base_height,
        }

    def _build_records(self, path: Path, label: int, category: str) -> list[SampleRecord]:
        image = self._prepare_base_image(self._read_image(path))
        height, width = image.shape[:2]

        if not self.patching_enabled:
            cache_path = self._build_cache_path(path, category, 0)
            if cache_path is not None and not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cache_path), self._finalize_patch(image))
            return [
                SampleRecord(
                    path=path,
                    label=label,
                    category=category,
                    patch_index=0,
                    patch_x=0,
                    patch_y=0,
                    patch_w=width,
                    patch_h=height,
                    base_width=width,
                    base_height=height,
                    cache_path=cache_path,
                )
            ]

        x_positions = self._sliding_positions(width, self.patch_size, self.patch_stride)
        y_positions = self._sliding_positions(height, self.patch_size, self.patch_stride)
        candidates: list[tuple[int, int, int, int]] = []
        for y in y_positions:
            for x in x_positions:
                patch_w = min(self.patch_size, width - x)
                patch_h = min(self.patch_size, height - y)
                patch = image[y : y + patch_h, x : x + patch_w]
                if float(np.std(patch)) < self.min_patch_std:
                    continue
                candidates.append((x, y, patch_w, patch_h))

        if not candidates:
            candidates.append((0, 0, width, height))

        if self.max_patches_per_image is not None and len(candidates) > self.max_patches_per_image:
            indices = np.linspace(0, len(candidates) - 1, self.max_patches_per_image, dtype=int)
            candidates = [candidates[int(idx)] for idx in indices]

        records: list[SampleRecord] = []
        for patch_index, (x, y, patch_w, patch_h) in enumerate(candidates):
            cache_path = self._build_cache_path(path, category, patch_index)
            if cache_path is not None and not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                patch = image[y : y + patch_h, x : x + patch_w]
                cv2.imwrite(str(cache_path), self._finalize_patch(patch))
            records.append(
                SampleRecord(
                    path=path,
                    label=label,
                    category=category,
                    patch_index=patch_index,
                    patch_x=x,
                    patch_y=y,
                    patch_w=patch_w,
                    patch_h=patch_h,
                    base_width=width,
                    base_height=height,
                    cache_path=cache_path,
                )
            )

        return records

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        return image

    def _prepare_base_image(self, image: np.ndarray) -> np.ndarray:
        prepared = image
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            prepared = prepared[y1:y2, x1:x2]

        if self.normalize_orientation and prepared.shape[0] > prepared.shape[1]:
            prepared = cv2.rotate(prepared, cv2.ROTATE_90_CLOCKWISE)

        if self.auto_crop:
            prepared = self._auto_crop_foreground(prepared)

        return prepared

    def _auto_crop_foreground(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 8, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)
        if w * h < 0.25 * image.shape[0] * image.shape[1]:
            return image

        pad = 12
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        return image[y1:y2, x1:x2]

    def _finalize_patch(self, image: np.ndarray) -> np.ndarray:
        patch = image
        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            patch = clahe.apply(patch)

        if self.gaussian_blur:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)

        if self.median_blur:
            patch = cv2.medianBlur(patch, 3)

        return self._resize_with_padding(patch)

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = self.image_size / float(max(height, width))
        resized = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

        fill_value = int(np.mean(resized)) if self.pad_mode == "mean" else 0
        canvas = np.full((self.image_size, self.image_size), fill_value, dtype=np.uint8)
        new_h, new_w = resized.shape[:2]
        top = (self.image_size - new_h) // 2
        left = (self.image_size - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if self.augment_mode == "none":
            return image
        if self.augment_mode == "spatial":
            return self._spatial_augment(image)
        if self.augment_mode == "frequency":
            return self._frequency_augment(image)
        raise ValueError(f"不支持的增强模式: {self.augment_mode}")

    def _spatial_augment(self, image: np.ndarray) -> np.ndarray:
        augmented = image.copy()
        if random.random() < 0.5:
            augmented = cv2.flip(augmented, 1)
        if random.random() < 0.3:
            augmented = cv2.flip(augmented, 0)

        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        scale = random.uniform(0.96, 1.04)
        center = (self.image_size / 2.0, self.image_size / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        augmented = cv2.warpAffine(
            augmented,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        alpha = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
        beta = random.uniform(-18.0, 18.0)
        augmented = np.clip(augmented.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return augmented

    def _frequency_augment(self, image: np.ndarray) -> np.ndarray:
        augmented = self._spatial_augment(image)
        image_float = augmented.astype(np.float32) / 255.0
        spectrum = np.fft.fft2(image_float)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        noise = np.random.normal(0.0, self.fft_noise_scale, size=image_float.shape).astype(np.float32)
        amplitude = amplitude * np.clip(1.0 + noise, 0.85, 1.15)
        perturbed = np.fft.ifft2(amplitude * np.exp(1j * phase)).real
        perturbed = np.clip(perturbed, 0.0, 1.0)
        return (perturbed * 255.0).astype(np.uint8)

    @staticmethod
    def _sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
        if length <= patch_size:
            return [0]

        positions = list(range(0, max(length - patch_size, 0) + 1, stride))
        last = length - patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _build_cache_path(self, path: Path, category: str, patch_index: int) -> Path | None:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        return self.cache_dir / self.split / category / f"{path.stem}_{patch_index:03d}.png"


def build_dataloaders_v2(config: dict[str, Any], max_test_samples: int | None = None) -> dict[str, DataLoader]:
    data_root = config["paths"]["data_root"]
    preprocess = config["preprocess"]
    patching = config.get("patching", {})
    augment = config["augment"]
    training = config["training"]
    loader_kwargs = resolve_dataloader_kwargs(training)

    common = {
        "data_root": data_root,
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
        "fft_noise_scale": augment.get("fft_noise_scale", 0.06),
        "rotation_deg": augment.get("rotation_deg", 6.0),
        "brightness_jitter": augment.get("brightness_jitter", 0.08),
    }

    datasets = {
        "train": SprayImageDatasetV2(
            split="train",
            augment_mode=augment.get("mode", "none"),
            **common,
        ),
        "val": SprayImageDatasetV2(
            split="val",
            augment_mode="none",
            **common,
        ),
        "test": SprayImageDatasetV2(
            split="test",
            augment_mode="none",
            max_samples=max_test_samples,
            **common,
        ),
    }

    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **loader_kwargs,
        ),
    }
