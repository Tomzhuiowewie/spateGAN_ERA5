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


def weighted_wet_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    wet_threshold: float = 0.1,
    wet_weight: float = 2.0,
) -> torch.Tensor:
    """湿区加权 L1，避免模型只优化大量无雨像素。"""

    weights = torch.ones_like(target)
    weights = torch.where(target > wet_threshold, weights * wet_weight, weights)
    return (weights * (prediction - target).abs()).mean()


def base_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """确定性基础预测损失。"""

    return F.l1_loss(prediction, target) + 0.1 * F.mse_loss(prediction, target)


def uncertainty_loss(
    uncertainty: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """让 u_map 学习基础预测误差尺度。"""

    return F.l1_loss(uncertainty, (target - prediction).abs().detach())


def diffusion_noise_loss(predicted_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """标准扩散噪声预测损失。"""

    return F.mse_loss(predicted_noise, noise)


def mass_conservation_loss(
    prediction: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """弱 ERA5 总量一致性约束。

    只比较 batch 内每个样本的平均降水量，避免硬缩放破坏局地结构。
    """

    era5_mean = x.sum(dim=1, keepdim=True).mean(dim=(2, 3, 4), keepdim=True) / 6.0
    pred_mean = prediction.mean(dim=(2, 3, 4), keepdim=True)
    return F.l1_loss(pred_mean, era5_mean.clamp_min(eps))


def pde_residual_loss(
    rain: torch.Tensor,
    v_field: torch.Tensor,
    s_field: torch.Tensor,
    p_rain: torch.Tensor,
    wet_threshold: float = 0.1,
) -> torch.Tensor:
    """降水平流-源汇方程的弱残差约束。"""

    if rain.shape[2] < 2 or rain.shape[3] < 2 or rain.shape[4] < 2:
        return rain.new_zeros(())

    dt = rain[:, :, 1:] - rain[:, :, :-1]
    rain_mid = rain[:, :, :-1]
    vx = v_field[:, 0:1, :-1]
    vy = v_field[:, 1:2, :-1]
    source = s_field[:, :, :-1]
    wet = ((rain_mid > wet_threshold) | (p_rain[:, :, :-1] > 0.5)).float()

    flux_x = vx * rain_mid
    flux_y = vy * rain_mid
    div_x = F.pad(flux_x[:, :, :, :, 1:] - flux_x[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))
    div_y = F.pad(flux_y[:, :, :, 1:, :] - flux_y[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0))
    residual = dt + div_x + div_y - source
    return F.huber_loss(residual * (1.0 + wet), torch.zeros_like(residual), delta=1.0)


def smoothness_loss(v_field: torch.Tensor, s_field: torch.Tensor) -> torch.Tensor:
    """约束学习到的速度场和源汇场不要过度振荡。"""

    loss = v_field.new_zeros(())
    for tensor in (v_field, s_field):
        loss = loss + (tensor[:, :, 1:] - tensor[:, :, :-1]).abs().mean()
        loss = loss + (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
        loss = loss + (tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]).abs().mean()
    return loss
