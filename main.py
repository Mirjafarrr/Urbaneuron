from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import get_dataloaders
from src.engine import fit
from src.losses import ComboLoss
from src.model import UNetPlusPlusResNeXt50
from src.utils import (
    compute_class_weights,
    load_config,
    load_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net++ (ResNeXt-50) on LoveDA 512x512 tiles."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--lr-override",
        type=str,
        default=None,
        help="Override LR after resume, e.g. '2.5e-5,2.5e-4' for encoder,decoder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    class_cfg = cfg.get("classes", {})

    data_dir = Path(args.data_dir or data_cfg.get("final_data", "data/final"))
    batch_size = args.batch_size or train_cfg.get("batch_size", 4)
    epochs = args.epochs or train_cfg.get("epochs", 100)
    num_workers = args.num_workers or train_cfg.get("num_workers", 4)
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.get("checkpoint_dir", "checkpoints"))
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = not args.no_amp and train_cfg.get("mixed_precision", True)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    num_classes = class_cfg.get("num_classes", 8)
    ignore_index = class_cfg.get("ignore_index", 0)

    encoder_lr = train_cfg.get("encoder_lr", 5e-5)
    decoder_lr = train_cfg.get("decoder_lr", 5e-4)
    loss_cfg = train_cfg.get("loss", {})
    ce_weight = loss_cfg.get("ce_weight", 0.7)
    dice_weight = loss_cfg.get("dice_weight", 0.3)
    sched_cfg = train_cfg.get("scheduler", {})
    seed = args.seed or train_cfg.get("random_seed", data_cfg.get("split", {}).get("random_seed", 42))
    log_interval = train_cfg.get("log_interval", 10)
    save_every = train_cfg.get("save_every", 1)

    print(f"Device: {device_str}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"Data dir: {data_dir}")
    print(f"Batch size: {batch_size}  |  Epochs: {epochs}  |  Workers: {num_workers}")
    print(f"Encoder LR: {encoder_lr}  |  Decoder LR: {decoder_lr}")
    print(f"Mixed precision: {mixed_precision}  |  Grad accum steps: {gradient_accumulation_steps}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    set_seed(seed)

    device = torch.device(device_str)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = UNetPlusPlusResNeXt50(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
    )
    model = model.to(device)

    print("\nComputing class weights from training set ...")
    class_weights = compute_class_weights(
        train_loader, num_classes=num_classes, ignore_index=ignore_index, device=device
    )
    print(f"Class weights: {class_weights.tolist()}")

    loss_fn = ComboLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ]
    )

    scheduler = None
    sched_type = sched_cfg.get("type", "")
    if sched_type == "plateaus":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
        print(f"LR scheduler: ReduceLROnPlateau (factor={scheduler.factor}, patience={scheduler.patience}, min_lr={scheduler.min_lrs})")
    elif sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", epochs),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
        print(f"LR scheduler: CosineAnnealing (T_max={scheduler.T_max}, eta_min={scheduler.eta_min})")

    start_epoch = 0
    best_metric = 0.0
    resume_from = Path(args.resume) if args.resume else None
    if resume_from is None:
        auto_resume = checkpoint_dir / "last.pt"
        if auto_resume.exists():
            print(f"Found checkpoint: {auto_resume} — auto-resuming")
            resume_from = auto_resume
    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")
        start_epoch, best_metric, ckpt = load_checkpoint(
            model, optimizer, resume_from, device
        )
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("  restored scheduler state")
            except (RuntimeError, KeyError, TypeError) as e:
                print(f"  skipping scheduler state restore (type changed): {e}")
        start_epoch += 1
        print(f"Resumed - start_epoch={start_epoch}, best_mIoU={best_metric:.4f}")

    if args.lr_override is not None:
        lr_values = [float(x.strip()) for x in args.lr_override.split(",")]
        if len(lr_values) != 2:
            print("ERROR: --lr-override requires two comma-separated values (encoder,decoder)")
            return
        enc_lr, dec_lr = lr_values
        optimizer.param_groups[0]["lr"] = enc_lr
        optimizer.param_groups[1]["lr"] = dec_lr
        print(f"LR overridden - encoder: {enc_lr:.2e}, decoder: {dec_lr:.2e}")

    print("\nStarting training ...\n")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        log_interval=log_interval,
        start_epoch=start_epoch,
        best_metric=best_metric,
        num_classes=num_classes,
        ignore_index=ignore_index,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()