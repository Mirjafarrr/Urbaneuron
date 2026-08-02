from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import IMAGENET_MEAN, IMAGENET_STD

NUM_CLASSES: int = 8
IGNORE_INDEX: int = 0

LOVEDA_CLASS_NAMES = {
    0: "ignore",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

IMG_EXTS = {".png", ".tif", ".tiff"}


def get_transforms(train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class LoveDADataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        transform: A.Compose | None = None,
    ):
        super().__init__()
        data_root = Path(data_root)
        self.img_dir = data_root / "images_png"
        self.mask_dir = data_root / "masks_png"

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        img_files = sorted(
            p
            for p in self.img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        self._paths: List[Tuple[Path, Path]] = []
        for img_path in img_files:
            mask_path = self.mask_dir / img_path.name
            if mask_path.is_file():
                self._paths.append((img_path, mask_path))

        if not self._paths:
            raise RuntimeError(
                f"No image/mask pairs found in {self.img_dir} / {self.mask_dir}"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._paths[idx]

        img: np.ndarray = self._read_rgb(img_path)
        mask: np.ndarray = self._read_mask(mask_path)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2, 0, 1))
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.ascontiguousarray(mask).copy()).long()

        return img, mask

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = Image.open(path)
        return np.asarray(mask, dtype=np.uint8)


def get_dataloaders(
    data_dir: str | Path = "data/final",
    batch_size: int = 8,
    num_workers: int = 0,
    train_transform: A.Compose | None = None,
    val_transform: A.Compose | None = None,
) -> Tuple[DataLoader, DataLoader]:
    if train_transform is None:
        train_transform = get_transforms(train=True)
    if val_transform is None:
        val_transform = get_transforms(train=False)

    train_ds = LoveDADataset(Path(data_dir) / "train", transform=train_transform)
    val_ds = LoveDADataset(Path(data_dir) / "val", transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader