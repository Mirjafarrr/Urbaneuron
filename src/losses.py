from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        ignore_index: int = 0,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
        smooth: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        ce = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)

        dice_scores: list[torch.Tensor] = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            pred_c = probs[:, c, ...]
            target_c = (target == c).to(pred_c.dtype)
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_c)

        dice = 1.0 - torch.stack(dice_scores).mean()

        return self.ce_weight * ce + self.dice_weight * dice