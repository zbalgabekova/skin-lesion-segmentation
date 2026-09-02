"""
Loss functions for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.training_config import (
    LOSS_FUNCTION,
    BCE_LOSS,
    DICE_LOSS,
    DICE_BCE_LOSS,
    FOCAL_LOSS,
)


# ==========================================================
# Dice Loss
# ==========================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """

        # Compute Dice in float32 for numerical stability
        probs = torch.sigmoid(logits).float()
        targets = targets.float()

        # Flatten each image separately
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probs * targets).sum(dim=1)

        union = (
            probs.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1.0 - dice.mean()


# ==========================================================
# Dice + BCE Loss
# ==========================================================

class DiceBCELoss(nn.Module):
    """
    Dice + BCE Loss.
    """

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        return bce_loss + dice_loss


# ==========================================================
# Binary Focal Loss
# ==========================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    """

    def __init__(
        self,
        alpha=0.8,
        gamma=2.0,
    ):
        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

    def forward(
        self,
        logits,
        targets,
    ):

        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-bce)

        focal = (
            self.alpha
            * (1 - pt) ** self.gamma
            * bce
        )

        return focal.mean()


# ==========================================================
# Factory
# ==========================================================

def create_loss(
    loss_name=LOSS_FUNCTION,
):
    """
    Create loss function.

    Args:
        loss_name:
            Name of the loss.

    Returns:
        nn.Module
    """

    loss_name = loss_name.lower()

    if loss_name == BCE_LOSS:
        return nn.BCEWithLogitsLoss()

    if loss_name == DICE_LOSS:
        return DiceLoss()

    if loss_name == DICE_BCE_LOSS:
        return DiceBCELoss()

    if loss_name == FOCAL_LOSS:
        return FocalLoss()

    raise ValueError(
        f"Unknown loss function: {loss_name}"
    )