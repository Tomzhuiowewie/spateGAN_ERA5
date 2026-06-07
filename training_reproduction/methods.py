"""训练方法定义。

`trainer.py` 只负责取数据和循环；这里负责具体怎么训练一步。
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from losses import (
    discriminator_loss,
    generator_adversarial_loss,
)
from model import CNNDownscaler, Discriminator, Generator


class TrainingMethod:
    """训练方法基类。每个训练方法都应该继承这个类，并实现下面的三个方法"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def autocast(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """训练一个 batch，并返回需要打印的 loss。"""
        raise NotImplementedError

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        raise NotImplementedError

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        raise NotImplementedError

    def checkpoint_state(self) -> dict:
        """返回完整 checkpoint 需要保存的内容。"""
        raise NotImplementedError

    def inference_state(self) -> dict[str, torch.Tensor]:
        """返回推理时需要加载的模型权重。"""
        raise NotImplementedError


class AdversarialGANMethod(TrainingMethod):
    """论文里的 cGAN 训练方式。"""

    def __init__(
        self,
        device: torch.device,
        generator_lr: float = 1e-4,
        discriminator_lr: float = 2e-4,
        l1_weight: float = 1.0,
        ensemble_size: int = 3,
    ) -> None:
        super().__init__(device)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.l1_weight = l1_weight
        self.ensemble_size = ensemble_size

        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(0.0, 0.999),
        )
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.0, 0.5),
        )

    def _next_dropout_seed(self) -> int:
        return random.randint(0, 2**31 - 1)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """先训练判别器，再训练生成器。"""

        d_loss = self._train_discriminator_step(x, y)
        g_loss, g_adv, g_l1 = self._train_generator_step(x, y)
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_adv": g_adv,
            "g_l1": g_l1,
        }

    def _train_discriminator_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """判别器学习区分真实降水和生成降水。"""

        self.discriminator.train()
        self.generator.train()

        self.d_optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            with torch.no_grad():
                fake = self.generator(x, self._next_dropout_seed())

            real_logits = self.discriminator(x, y)
            fake_logits = self.discriminator(x, fake)
            loss = discriminator_loss(real_logits, fake_logits)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.d_optimizer)
        self.scaler.update()
        return float(loss.detach().cpu())

    def _train_generator_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
        """生成器同时优化对抗损失和 ensemble L1 损失"""

        self.discriminator.train()
        self.generator.train()

        self.g_optimizer.zero_grad(set_to_none=True)
        seeds = [self._next_dropout_seed() for _ in range(self.ensemble_size)]

        # The ensemble loss depends on the mean prediction. Compute its exact
        # output gradient first, then recompute and backpropagate one member at
        # a time so only one very large generator graph resides on the GPU.
        with torch.no_grad(), self.autocast():
            ensemble_sum = torch.zeros_like(y)
            for seed in seeds:
                ensemble_sum.add_(self.generator(x, seed))
            ensemble_mean = ensemble_sum / self.ensemble_size
            l1 = F.l1_loss(ensemble_mean, y)
            l1_output_grad = (
                torch.sign(ensemble_mean - y)
                * (self.l1_weight / (self.ensemble_size * y.numel()))
            )

        discriminator_requires_grad = [
            parameter.requires_grad for parameter in self.discriminator.parameters()
        ]
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)

        adv = torch.zeros((), device=self.device)
        try:
            for member_index, seed in enumerate(seeds):
                with self.autocast():
                    prediction = self.generator(x, seed)
                    member_loss = (prediction * l1_output_grad).sum()
                    if member_index == 0:
                        fake_logits = self.discriminator(x, prediction)
                        adv = generator_adversarial_loss(fake_logits)
                        member_loss = member_loss + adv
                self.scaler.scale(member_loss).backward()
        finally:
            for parameter, requires_grad in zip(
                self.discriminator.parameters(), discriminator_requires_grad
            ):
                parameter.requires_grad_(requires_grad)

        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        loss = adv.detach() + self.l1_weight * l1
        return (
            float(loss.detach().cpu()),
            float(adv.detach().cpu()),
            float(l1.detach().cpu()),
        )

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        self.generator.eval()
        with torch.no_grad(), self.autocast():
            ensemble_sum = torch.zeros_like(y)
            for member_index in range(self.ensemble_size):
                ensemble_sum.add_(self.generator(x, seed + member_index))
            prediction = ensemble_sum / self.ensemble_size
            mae = F.l1_loss(prediction, y)
            mse = F.mse_loss(prediction, y)
        return {"val_mae": float(mae.cpu()), "val_mse": float(mse.cpu())}, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            return self.generator(torch.clamp(x, min=0.0), seed)

    def checkpoint_state(self) -> dict:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_optimizer": self.g_optimizer.state_dict(),
            "discriminator_optimizer": self.d_optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return self.generator.state_dict()


class CNNMethod(TrainingMethod):
    """普通 CNN + L1 损失的监督训练方式。"""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-4,
    ) -> None:
        super().__init__(device)
        self.model = CNNDownscaler().to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.0, 0.999),
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            prediction = self.model(x)
            loss = F.l1_loss(prediction, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {"loss": float(loss.detach().cpu())}

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        del seed
        self.model.eval()
        with torch.no_grad(), self.autocast():
            prediction = self.model(x)
            mae = F.l1_loss(prediction, y)
            mse = F.mse_loss(prediction, y)
        return {"val_mae": float(mae.cpu()), "val_mse": float(mse.cpu())}, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        del seed
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.clamp(x, min=0.0))

    def checkpoint_state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()


def build_training_method(method_name: str, device: torch.device, config) -> TrainingMethod:
    """根据配置创建训练方法"""

    if method_name == "gan":
        return AdversarialGANMethod(
            device=device,
            generator_lr=config.generator_lr,
            discriminator_lr=config.discriminator_lr,
            l1_weight=config.l1_weight,
            ensemble_size=config.ensemble_size,
        )
    if method_name == "cnn":
        return CNNMethod(
            device=device,
            lr=config.generator_lr,
        )
    raise ValueError(f"未知 training_method: {method_name}")
