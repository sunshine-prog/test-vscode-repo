from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    with target.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def configure_reproducibility(seed: int, reproducibility: dict[str, Any] | None = None) -> dict[str, Any]:
    reproducibility = reproducibility or {}
    deterministic = bool(reproducibility.get("deterministic", False))
    warn_only = bool(reproducibility.get("warn_only", True))
    cublas_workspace_config = reproducibility.get("cublas_workspace_config", ":4096:8")

    if deterministic and cublas_workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_workspace_config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.use_deterministic_algorithms(False)

    return {
        "seed": int(seed),
        "deterministic": deterministic,
        "warn_only": warn_only,
    }


def set_seed(seed: int) -> None:
    configure_reproducibility(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a GPU-enabled PyTorch build or switch device to cpu/auto.")
    return torch.device(device_name)


def _parse_auto_bool(value: Any, *, auto_value: bool, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return auto_value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return auto_value
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Unsupported value for training.{option_name}: {value!r}")


def _parse_num_workers(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        num_workers = value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            cpu_count = os.cpu_count() or 1
            # Windows multiprocessing can be fragile in student environments,
            # so keep the default conservative there.
            return 0 if os.name == "nt" else max(1, min(4, cpu_count - 1))
        num_workers = int(normalized)
    else:
        num_workers = int(value)

    if num_workers < 0:
        raise ValueError(f"training.num_workers must be >= 0, got {num_workers}")
    return num_workers


def resolve_dataloader_kwargs(training: dict[str, Any]) -> dict[str, Any]:
    num_workers = _parse_num_workers(training.get("num_workers", 0))
    pin_memory = _parse_auto_bool(
        training.get("pin_memory", False),
        auto_value=torch.cuda.is_available(),
        option_name="pin_memory",
    )
    persistent_workers = _parse_auto_bool(
        training.get("persistent_workers", False),
        auto_value=num_workers > 0,
        option_name="persistent_workers",
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(training["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        prefetch_factor = training.get("prefetch_factor")
        if prefetch_factor is not None:
            if isinstance(prefetch_factor, str) and prefetch_factor.strip().lower() == "auto":
                loader_kwargs["prefetch_factor"] = 2
            else:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return loader_kwargs


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    scheduler_config = training.get("scheduler", {})
    scheduler_type = str(scheduler_config.get("type", "cosine")).strip().lower()

    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.01))
        eta_min = float(training["learning_rate"]) * min_lr_ratio
        return CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=eta_min)
    if scheduler_type == "step":
        step_size = int(scheduler_config.get("step_size", max(1, int(epochs) // 2)))
        gamma = float(scheduler_config.get("gamma", 0.5))
        return StepLR(optimizer, step_size=max(step_size, 1), gamma=gamma)

    raise ValueError(f"Unsupported training.scheduler.type: {scheduler_type!r}")
