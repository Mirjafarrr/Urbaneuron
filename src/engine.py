from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import ComboLoss
from .metrics import SegmentationMetrics
from .utils import save_checkpoint


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: ComboLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    epochs: int = 100,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    checkpoint_dir: str | Path = "checkpoints",
    save_every: int = 10,
    log_interval: int = 10,
    start_epoch: int = 0,
    best_metric: float = 0.0,
    num_classes: int = 8,
    ignore_index: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.get_device_properties(device).major >= 8:
            print("TF32 enabled (Ampere+ GPU detected)")

        gpu_name = torch.cuda.get_device_name(device)
        vram_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU: {gpu_name}  |  VRAM: {vram_total_gb:.1f} GB")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler("cuda") if mixed_precision and device.type == "cuda" else None

    val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [train]",
            unit="batch",
            file=sys.stdout,
        )
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = loss_fn(logits, masks) / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(images)
                loss = loss_fn(logits, masks) / gradient_accumulation_steps
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % log_interval == 0:
                avg = running_loss / (batch_idx + 1)
                pbar.set_postfix(loss=f"{avg:.4f}")

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_metrics.reset()

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val ]", unit="batch", file=sys.stdout
            ):
                images = images.to(device)
                masks = masks.to(device)

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)

                val_metrics.update(logits.cpu(), masks.cpu())

        results = val_metrics.compute()
        current_miou = results["mIoU"]

        per_class = results["per_class_IoU"]
        class_names = {
            0: "ignore", 1: "bkg", 2: "bldg", 3: "road",
            4: "water", 5: "barren", 6: "forest", 7: "agri",
        }
        iou_str = "  ".join(
            f"{class_names.get(c, c)}:{per_class[c]:.3f}"
            for c in range(num_classes)
        )

        epoch_secs = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_mIoU={current_miou:.4f}  "
            f"val_acc={results['pixel_accuracy']:.4f}  "
            f"time={epoch_secs:.1f}s ({epoch_secs/60:.1f}m)"
        )
        print(f"  per-class IoU: {iou_str}")

        if current_miou > best_metric:
            best_metric = current_miou
            save_checkpoint(
                model, optimizer, epoch, best_metric, results,
                checkpoint_dir / "best.pt",
            )
            print(f"  -> new best model saved (mIoU {best_metric:.4f})")

        save_checkpoint(
            model, optimizer, epoch, best_metric, results,
            checkpoint_dir / "last.pt",
            scheduler=scheduler,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_miou)
                current_lr = [pg["lr"] for pg in optimizer.param_groups]
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()
            print(f"  LR after step: {[f'{lr:.2e}' for lr in current_lr]}")

    print(f"Training finished. Best val mIoU: {best_metric:.4f}")
