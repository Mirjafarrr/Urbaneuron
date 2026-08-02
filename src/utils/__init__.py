from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    metrics: dict,
    path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[int, float, dict[str, Any]]:
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        ckpt.get("epoch", 0),
        ckpt.get("best_metric", 0.0),
        ckpt,
    )


def compute_class_weights(
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 8,
    ignore_index: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.int64)

    for _, masks in dataloader:
        flat = masks.flatten().long()
        counts += torch.bincount(flat, minlength=num_classes)

    weights = torch.zeros(num_classes)
    valid_classes = [c for c in range(num_classes) if c != ignore_index]
    total_valid_pixels = sum(counts[c].item() for c in valid_classes)

    for c in valid_classes:
        if counts[c] > 0:
            weights[c] = total_valid_pixels / (len(valid_classes) * counts[c].item())

    weights[ignore_index] = 0.0
    return weights.to(device)