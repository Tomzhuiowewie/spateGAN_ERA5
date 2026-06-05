"""训练方法定义。

`trainer.py` 只负责取数据和循环；这里负责具体怎么训练一步。
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from losses import (
    discriminator_loss,
    ensemble_l1_loss,
    generator_adversarial_loss,
)
from model import CNNDownscaler, Discriminator, Generator


class TrainingMethod:
    """训练方法基类。每个训练方法都应该继承这个类，并实现下面的三个方法"""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """训练一个 batch，并返回需要打印的 loss。"""
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
        with torch.no_grad():
            fake = self.generator(x, self._next_dropout_seed())

        real_logits = self.discriminator(x, y)
        fake_logits = self.discriminator(x, fake.detach())
        loss = discriminator_loss(real_logits, fake_logits)
        loss.backward()
        self.d_optimizer.step()
        return float(loss.detach().cpu())

    def _train_generator_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
        """生成器同时优化对抗损失和 ensemble L1 损失"""

        self.discriminator.train()
        self.generator.train()

        self.g_optimizer.zero_grad(set_to_none=True)
        predictions = [
            self.generator(x, self._next_dropout_seed())
            for _ in range(self.ensemble_size)
        ]
        fake_logits = self.discriminator(x, predictions[0])
        adv = generator_adversarial_loss(fake_logits)
        l1 = ensemble_l1_loss(predictions, y)
        loss = adv + self.l1_weight * l1
        loss.backward()
        self.g_optimizer.step()
        return (
            float(loss.detach().cpu()),
            float(adv.detach().cpu()),
            float(l1.detach().cpu()),
        )

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

        prediction = self.model(x)
        loss = F.l1_loss(prediction, y)

        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

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
