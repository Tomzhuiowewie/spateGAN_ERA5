"""Loss functions for the spateGAN-ERA5 training reproduction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy discriminator loss."""

    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real_loss + fake_loss


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator BCE loss that tries to classify generated fields as real."""

    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def ensemble_l1_loss(predictions: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """L1 loss between target and the mean of an ensemble of predictions."""

    if not predictions:
        raise ValueError("predictions must contain at least one tensor")
    ensemble_mean = torch.stack(predictions, dim=0).mean(dim=0)
    return F.l1_loss(ensemble_mean, target)
