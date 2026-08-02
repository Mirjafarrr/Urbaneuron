from __future__ import annotations

import torch
import torch.nn.functional as F


class SegmentationMetrics:
    def __init__(self, num_classes: int = 8, ignore_index: int = 0) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._cm = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        mask = target != self.ignore_index

        p = preds[mask].long()
        t = target[mask].long()

        indices = t * self.num_classes + p
        cm_batch = torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

        self._cm = self._cm + cm_batch

    def compute(self) -> dict:
        cm = self._cm.float()

        diag = torch.diag(cm)
        row_sum = cm.sum(dim=1)
        col_sum = cm.sum(dim=0)
        union = row_sum + col_sum - diag

        iou_per_class = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            if union[c] > 0:
                iou_per_class[c] = (diag[c] / union[c]).item()

        valid_classes = [c for c in range(self.num_classes) if c != self.ignore_index]
        miou = (
            torch.tensor([iou_per_class[c] for c in valid_classes]).mean().item()
        )

        total_correct = diag.sum().item()
        total_pixels = cm.sum().item()
        pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

        return {
            "mIoU": miou,
            "per_class_IoU": iou_per_class.tolist(),
            "pixel_accuracy": pixel_acc,
        }