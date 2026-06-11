"""训练方法定义。

`trainer.py` 只负责取数据和循环；这里负责具体怎么训练一步。
"""

from __future__ import annotations

import random

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from losses import (
    base_reconstruction_loss,
    diffusion_noise_loss,
    discriminator_loss,
    generator_adversarial_loss,
    mass_conservation_loss,
    pde_residual_loss,
    smoothness_loss,
    uncertainty_loss,
    weighted_wet_l1_loss,
)
from model import CNNDownscaler, Discriminator, Generator, PGBaseNet, PGERDDownscaler


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


def _wrap_ddp(model: torch.nn.Module, device: torch.device, use_ddp: bool) -> torch.nn.Module:
    """需要多卡训练时，把模型交给 PyTorch DDP 管理。"""

    if not use_ddp:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index])
    return DistributedDataParallel(model)


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    """保存权重时取出 DDP 里面真正的模型。"""

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _common_regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    wet_threshold: float = 0.1,
) -> dict[str, float]:
    mae = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    rmse = torch.sqrt(mse)
    bias = (prediction - target).mean()
    wet_mask = target > wet_threshold
    wet_mae = (prediction[wet_mask] - target[wet_mask]).abs().mean() if wet_mask.any() else mae
    return {
        "val_mae": float(mae.cpu()),
        "val_mse": float(mse.cpu()),
        "val_rmse": float(rmse.cpu()),
        "val_bias": float(bias.cpu()),
        "val_wet_mae": float(wet_mae.cpu()),
    }


class ERA5InterpolationMethod(TrainingMethod):
    """无训练 ERA5 插值基线。"""

    def __init__(
        self,
        device: torch.device,
        mode: str = "trilinear",
        wet_threshold: float = 0.1,
    ) -> None:
        super().__init__(device)
        if mode not in {"trilinear", "nearest"}:
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        self.mode = mode
        self.wet_threshold = wet_threshold

    def _predict(self, x: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        coarse_total = x.sum(dim=1, keepdim=True).clamp_min(0.0)
        if self.mode == "nearest":
            return F.interpolate(coarse_total, size=target_shape, mode="nearest")
        return F.interpolate(
            coarse_total,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            prediction = self._predict(x, y.shape[2:])
            loss = F.l1_loss(prediction, y)
        return {"loss": float(loss.cpu())}

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        del seed
        with torch.no_grad():
            prediction = self._predict(x, y.shape[2:])
            metrics = _common_regression_metrics(
                prediction,
                y,
                wet_threshold=self.wet_threshold,
            )
        return metrics, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        del seed
        return self._predict(torch.clamp(x, min=0.0), (48, 168, 168))

    def checkpoint_state(self) -> dict:
        return {"interpolation_mode": self.mode}

    def inference_state(self) -> dict[str, torch.Tensor]:
        return {}


class AdversarialGANMethod(TrainingMethod):
    """论文里的 cGAN 训练方式。"""

    def __init__(
        self,
        device: torch.device,
        generator_lr: float = 1e-4,
        discriminator_lr: float = 2e-4,
        l1_weight: float = 1.0,
        ensemble_size: int = 3,
        use_ddp: bool = False,
    ) -> None:
        super().__init__(device)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.generator = _wrap_ddp(self.generator, device, use_ddp)
        self.discriminator = _wrap_ddp(self.discriminator, device, use_ddp)
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
        generator = _raw_model(self.generator)
        generator.eval()
        with torch.no_grad(), self.autocast():
            ensemble_sum = torch.zeros_like(y)
            for member_index in range(self.ensemble_size):
                ensemble_sum.add_(generator(x, seed + member_index))
            prediction = ensemble_sum / self.ensemble_size
            mae = F.l1_loss(prediction, y)
            mse = F.mse_loss(prediction, y)
        return {"val_mae": float(mae.cpu()), "val_mse": float(mse.cpu())}, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        generator = _raw_model(self.generator)
        generator.eval()
        with torch.no_grad():
            return generator(torch.clamp(x, min=0.0), seed)

    def checkpoint_state(self) -> dict:
        return {
            "generator": _raw_model(self.generator).state_dict(),
            "discriminator": _raw_model(self.discriminator).state_dict(),
            "generator_optimizer": self.g_optimizer.state_dict(),
            "discriminator_optimizer": self.d_optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return _raw_model(self.generator).state_dict()


class CNNMethod(TrainingMethod):
    """普通 CNN + L1 损失的监督训练方式。"""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-4,
        use_ddp: bool = False,
    ) -> None:
        super().__init__(device)
        self.model = CNNDownscaler().to(device)
        self.model = _wrap_ddp(self.model, device, use_ddp)
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
        model = _raw_model(self.model)
        model.eval()
        with torch.no_grad(), self.autocast():
            prediction = model(x)
            metrics = _common_regression_metrics(prediction, y)
        return metrics, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        del seed
        model = _raw_model(self.model)
        model.eval()
        with torch.no_grad():
            return model(torch.clamp(x, min=0.0))

    def checkpoint_state(self) -> dict:
        return {
            "model": _raw_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return _raw_model(self.model).state_dict()


class STResUNetMethod(TrainingMethod):
    """只训练 PG-ERD 的确定性基础网络，用作强监督基线。"""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-4,
        base_channels: int = 32,
        wet_threshold: float = 0.1,
        use_ddp: bool = False,
    ) -> None:
        super().__init__(device)
        self.model = PGBaseNet(base_channels=base_channels).to(device)
        self.model = _wrap_ddp(self.model, device, use_ddp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.0, 0.999),
        )
        self.wet_threshold = wet_threshold

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            outputs = self.model(x)
            prediction = outputs["y_base"]
            base_loss = base_reconstruction_loss(prediction, y)
            recon_loss = weighted_wet_l1_loss(
                prediction,
                y,
                wet_threshold=self.wet_threshold,
            )
            mass_loss = mass_conservation_loss(prediction, x)
            loss = base_loss + 0.1 * recon_loss + 0.05 * mass_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            "loss": float(loss.detach().cpu()),
            "base": float(base_loss.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "mass": float(mass_loss.detach().cpu()),
        }

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        del seed
        model = _raw_model(self.model)
        model.eval()
        with torch.no_grad(), self.autocast():
            prediction = model(x)["y_base"]
            metrics = _common_regression_metrics(
                prediction,
                y,
                wet_threshold=self.wet_threshold,
            )
        return metrics, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        del seed
        model = _raw_model(self.model)
        model.eval()
        with torch.no_grad():
            return model(torch.clamp(x, min=0.0))["y_base"]

    def checkpoint_state(self) -> dict:
        return {
            "model": _raw_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return _raw_model(self.model).state_dict()


class PGERDMethod(TrainingMethod):
    """物理引导误差残差扩散训练方式。"""

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-4,
        base_channels: int = 32,
        diffusion_steps: int = 1000,
        sampling_steps: int = 5,
        ensemble_size: int = 3,
        base_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        diffusion_weight: float = 1.0,
        recon_weight: float = 0.1,
        mass_weight: float = 0.05,
        pde_weight: float = 0.01,
        smooth_weight: float = 0.005,
        wet_threshold: float = 0.1,
        use_ddp: bool = False,
    ) -> None:
        super().__init__(device)
        self.model = PGERDDownscaler(
            base_channels=base_channels,
            diffusion_steps=diffusion_steps,
        ).to(device)
        self.model = _wrap_ddp(self.model, device, use_ddp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.0, 0.999),
        )
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.ensemble_size = ensemble_size
        self.base_weight = base_weight
        self.uncertainty_weight = uncertainty_weight
        self.diffusion_weight = diffusion_weight
        self.recon_weight = recon_weight
        self.mass_weight = mass_weight
        self.pde_weight = pde_weight
        self.smooth_weight = smooth_weight
        self.wet_threshold = wet_threshold
        self.eps = 1e-4

        betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def _alpha_bar(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars[timestep].view(-1, 1, 1, 1, 1)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)

    def _q_sample(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self._alpha_bar(timestep)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def _predict_clean(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self._alpha_bar(timestep)
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-6)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self.autocast():
            raw_model = _raw_model(self.model)
            with torch.no_grad():
                detached_base = raw_model.base_forward(x)
                normalized_residual = (
                    (y - detached_base["y_base"]) / detached_base["u_map"].clamp_min(self.eps)
                ).clamp(-8.0, 8.0)
            timestep = self._sample_timesteps(x.shape[0])
            noise = torch.randn_like(normalized_residual)
            noisy_residual = self._q_sample(normalized_residual, timestep, noise)
            model_outputs = self.model(x, noisy_residual, timestep)
            y_base = model_outputs["y_base"]
            u_map = model_outputs["u_map"]
            predicted_noise = model_outputs["predicted_noise"]
            clean_residual = self._predict_clean(noisy_residual, timestep, predicted_noise).clamp(-8.0, 8.0)
            prediction = torch.clamp(y_base + u_map * clean_residual, min=0.0)

            base_loss = base_reconstruction_loss(y_base, y)
            unc_loss = uncertainty_loss(u_map, y_base, y)
            diff_loss = diffusion_noise_loss(predicted_noise, noise)
            recon_loss = weighted_wet_l1_loss(prediction, y, wet_threshold=self.wet_threshold)
            mass_loss = mass_conservation_loss(prediction, x)
            pde_loss = pde_residual_loss(
                prediction,
                model_outputs["v_field"],
                model_outputs["s_field"],
                model_outputs["p_rain"],
                wet_threshold=self.wet_threshold,
            )
            smooth_loss = smoothness_loss(model_outputs["v_field"], model_outputs["s_field"])
            loss = (
                self.base_weight * base_loss
                + self.uncertainty_weight * unc_loss
                + self.diffusion_weight * diff_loss
                + self.recon_weight * recon_loss
                + self.mass_weight * mass_loss
                + self.pde_weight * pde_loss
                + self.smooth_weight * smooth_loss
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            "loss": float(loss.detach().cpu()),
            "base": float(base_loss.detach().cpu()),
            "diff": float(diff_loss.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "mass": float(mass_loss.detach().cpu()),
            "pde": float(pde_loss.detach().cpu()),
        }

    def _ddim_sample(
        self,
        x: torch.Tensor,
        seed: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = _raw_model(self.model)
        model.eval()
        generator = torch.Generator(device=x.device).manual_seed(seed)
        base_outputs = model.base_forward(x)
        residual = torch.randn(
            base_outputs["y_base"].shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            self.sampling_steps,
            device=x.device,
        ).long()
        for index, timestep_value in enumerate(timesteps):
            timestep = timestep_value.repeat(x.shape[0])
            predicted_noise = model.predict_noise(x, residual, timestep, base_outputs)
            clean = self._predict_clean(residual, timestep, predicted_noise).clamp(-8.0, 8.0)
            if index == len(timesteps) - 1:
                residual = clean
            else:
                next_timestep = timesteps[index + 1].repeat(x.shape[0])
                next_alpha = self._alpha_bar(next_timestep)
                residual = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * predicted_noise
        prediction = torch.clamp(base_outputs["y_base"] + base_outputs["u_map"] * residual, min=0.0)
        return prediction, base_outputs

    @staticmethod
    def _crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        first = (samples - target.unsqueeze(0)).abs().mean()
        pairwise = (samples[:, None] - samples[None]).abs().mean()
        return first - 0.5 * pairwise

    def validation_step(
        self, x: torch.Tensor, y: torch.Tensor, seed: int
    ) -> tuple[dict[str, float], torch.Tensor]:
        predictions = []
        last_base_outputs = None
        with torch.no_grad(), self.autocast():
            for member_index in range(self.ensemble_size):
                prediction, base_outputs = self._ddim_sample(x, seed + member_index)
                predictions.append(prediction)
                last_base_outputs = base_outputs
            samples = torch.stack(predictions, dim=0)
            prediction = samples.mean(dim=0)
            mae = F.l1_loss(prediction, y)
            mse = F.mse_loss(prediction, y)
            rmse = torch.sqrt(mse)
            bias = (prediction - y).mean()
            wet_mask = y > self.wet_threshold
            wet_mae = (prediction[wet_mask] - y[wet_mask]).abs().mean() if wet_mask.any() else mae
            crps = self._crps(samples, y)
            pde = pde_residual_loss(
                prediction,
                last_base_outputs["v_field"],
                last_base_outputs["s_field"],
                last_base_outputs["p_rain"],
                wet_threshold=self.wet_threshold,
            )
        return {
            "val_mae": float(mae.cpu()),
            "val_mse": float(mse.cpu()),
            "val_rmse": float(rmse.cpu()),
            "val_bias": float(bias.cpu()),
            "val_wet_mae": float(wet_mae.cpu()),
            "val_crps": float(crps.cpu()),
            "val_pde": float(pde.cpu()),
        }, prediction

    def visualization_prediction(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        with torch.no_grad():
            prediction, _ = self._ddim_sample(torch.clamp(x, min=0.0), seed)
        return prediction

    def checkpoint_state(self) -> dict:
        return {
            "model": _raw_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def inference_state(self) -> dict[str, torch.Tensor]:
        return _raw_model(self.model).state_dict()


def build_training_method(method_name: str, device: torch.device, config) -> TrainingMethod:
    """根据配置创建训练方法"""

    if method_name == "era5_trilinear":
        return ERA5InterpolationMethod(
            device=device,
            mode="trilinear",
            wet_threshold=config.wet_threshold,
        )
    if method_name == "era5_nearest":
        return ERA5InterpolationMethod(
            device=device,
            mode="nearest",
            wet_threshold=config.wet_threshold,
        )
    if method_name == "gan":
        return AdversarialGANMethod(
            device=device,
            generator_lr=config.generator_lr,
            discriminator_lr=config.discriminator_lr,
            l1_weight=config.l1_weight,
            ensemble_size=config.ensemble_size,
            use_ddp=getattr(config, "distributed", False),
        )
    if method_name == "cnn":
        return CNNMethod(
            device=device,
            lr=config.generator_lr,
            use_ddp=getattr(config, "distributed", False),
        )
    if method_name == "st_resunet":
        return STResUNetMethod(
            device=device,
            lr=config.generator_lr,
            base_channels=config.base_channels,
            wet_threshold=config.wet_threshold,
            use_ddp=getattr(config, "distributed", False),
        )
    if method_name == "pg_erd":
        return PGERDMethod(
            device=device,
            lr=config.generator_lr,
            base_channels=config.base_channels,
            diffusion_steps=config.diffusion_steps,
            sampling_steps=config.sampling_steps,
            ensemble_size=config.ensemble_size,
            base_weight=config.base_weight,
            uncertainty_weight=config.uncertainty_weight,
            diffusion_weight=config.diffusion_weight,
            recon_weight=config.recon_weight,
            mass_weight=config.mass_weight,
            pde_weight=config.pde_weight,
            smooth_weight=config.smooth_weight,
            wet_threshold=config.wet_threshold,
            use_ddp=getattr(config, "distributed", False),
        )
    raise ValueError(f"未知 training_method: {method_name}")
