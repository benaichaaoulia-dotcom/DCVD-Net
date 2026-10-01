"""Binary cross-entropy and soft-Dice objective used by the experiment."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Global soft-Dice loss with the experiment's smoothing constant."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_flat = prediction.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = torch.sum(prediction_flat * target_flat)
        dice = (2.0 * intersection + self.smooth) / (
            torch.sum(prediction_flat) + torch.sum(target_flat) + self.smooth
        )
        return 1.0 - dice


def bce_dice_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Equal-weight BCE + Dice objective, with the recorded probability clamp."""
    bce = nn.functional.binary_cross_entropy(prediction.clamp(min=1e-7, max=1.0 - 1e-7), target)
    return bce + DiceLoss(smooth=1e-6)(prediction, target)
