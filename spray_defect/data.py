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


class SprayImageDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int = 128,
        roi: list[int] | None = None,
        clahe: bool = True,
        gaussian_blur: bool = True,
        median_blur: bool = True,
        pad_mode: str = "mean",
        augment_mode: str = "none",
        augment_methods: list[str] | None = None,
        fft_noise_scale: float = 0.06,
        rotation_deg: float = 6.0,
        brightness_jitter: float = 0.08,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.roi = roi
        self.clahe = clahe
        self.gaussian_blur = gaussian_blur
        self.median_blur = median_blur
        self.pad_mode = pad_mode
        self.augment_mode = augment_mode
        self.augment_methods = [str(method).strip().lower() for method in augment_methods] if augment_methods else None
        self.fft_noise_scale = fft_noise_scale
        self.rotation_deg = rotation_deg
        self.brightness_jitter = brightness_jitter

        categories = ["normal"] if split in {"train", "val"} else ["normal", "defect"]
        self.samples: list[SampleRecord] = []
        for category in categories:
            category_dir = self.root / split / category
            if not category_dir.exists():
                continue
            label = 1 if category == "defect" else 0
            for path in sorted(category_dir.glob("*.jpg")):
                self.samples.append(SampleRecord(path=path, label=label, category=category))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError(f"在 {self.root / split} 下没有找到可用的图片。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._read_image(sample.path)
        image = self._preprocess(image)
        if self.split == "train":
            image = self._augment(image)

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return {
            "image": tensor,
            "label": sample.label,
            "path": str(sample.path),
            "sample_id": sample.path.stem,
            "category": sample.category,
        }

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        return image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            image = image[y1:y2, x1:x2]

        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        if self.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        if self.median_blur:
            image = cv2.medianBlur(image, 3)

        return self._resize_with_padding(image)

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
        if self.augment_methods is not None:
            augmented = image
            for method in self.augment_methods:
                if method == "spatial":
                    augmented = self._spatial_only_augment(augmented)
                elif method == "photometric":
                    augmented = self._photometric_augment(augmented)
                elif method == "frequency":
                    augmented = self._frequency_only_augment(augmented)
                else:
                    raise ValueError(f"Unsupported augmentation method: {method}")
            return augmented
        if self.augment_mode == "none":
            return image
        if self.augment_mode == "photometric":
            return self._photometric_augment(image)
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

    def _spatial_only_augment(self, image: np.ndarray) -> np.ndarray:
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
        return augmented

    def _photometric_augment(self, image: np.ndarray) -> np.ndarray:
        augmented = image.copy()
        alpha = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
        beta = random.uniform(-18.0, 18.0)
        augmented = np.clip(augmented.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return augmented

    def _frequency_only_augment(self, image: np.ndarray) -> np.ndarray:
        image_float = image.astype(np.float32) / 255.0
        spectrum = np.fft.fft2(image_float)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        noise = np.random.normal(0.0, self.fft_noise_scale, size=image_float.shape).astype(np.float32)
        amplitude = amplitude * np.clip(1.0 + noise, 0.85, 1.15)
        perturbed = np.fft.ifft2(amplitude * np.exp(1j * phase)).real
        perturbed = np.clip(perturbed, 0.0, 1.0)
        return (perturbed * 255.0).astype(np.uint8)

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


def build_dataloaders(config: dict[str, Any], max_test_samples: int | None = None) -> dict[str, DataLoader]:
    data_root = config["paths"]["data_root"]
    preprocess = config["preprocess"]
    augment = config["augment"]
    training = config["training"]
    loader_kwargs = resolve_dataloader_kwargs(training)

    common = {
        "data_root": data_root,
        "image_size": preprocess["image_size"],
        "roi": preprocess.get("roi"),
        "clahe": preprocess.get("clahe", True),
        "gaussian_blur": preprocess.get("gaussian_blur", True),
        "median_blur": preprocess.get("median_blur", True),
        "pad_mode": preprocess.get("pad_mode", "mean"),
        "augment_methods": augment.get("methods"),
        "fft_noise_scale": augment.get("fft_noise_scale", 0.06),
        "rotation_deg": augment.get("rotation_deg", 6.0),
        "brightness_jitter": augment.get("brightness_jitter", 0.08),
    }

    datasets = {
        "train": SprayImageDataset(
            split="train",
            augment_mode=augment.get("mode", "none"),
            **common,
        ),
        "val": SprayImageDataset(
            split="val",
            augment_mode="none",
            **common,
        ),
        "test": SprayImageDataset(
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
