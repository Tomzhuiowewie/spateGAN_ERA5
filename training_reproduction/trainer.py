"""Training loop for the paper reproduction and compatible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from dataset import PairedPatchDataset
from methods import build_training_method


@dataclass
class TrainingConfig:
    x_path: str
    y_path: str
    x_pattern: str = "x*.nc"
    y_pattern: str = "y*.nc"
    y_variable: str | None = None
    x_target_shape: tuple[int, int, int] = (16, 28, 28)
    y_target_shape: tuple[int, int, int] = (48, 168, 168)
    training_method: str = "gan"
    output_dir: str = "training_reproduction/checkpoints"
    batch_size: int = 9
    max_steps: int = 200_000
    checkpoint_interval: int = 250
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    load_into_memory: bool = False
    generator_lr: float = 1e-4
    discriminator_lr: float = 2e-4
    l1_weight: float = 1.0
    ensemble_size: int = 3


class Trainer:
    """Outer training loop independent of the concrete optimization method."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self.dataset = PairedPatchDataset(
            config.x_path,
            config.y_path,
            x_pattern=config.x_pattern,
            y_pattern=config.y_pattern,
            x_target_shape=config.x_target_shape,
            y_variable=config.y_variable,
            y_target_shape=config.y_target_shape,
            load_into_memory=config.load_into_memory,
            seed=config.seed,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )

        self.method = build_training_method(config.training_method, self.device, config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        step = 0
        while step < self.config.max_steps:
            for x, y in self.loader:
                step += 1
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logs = self.method.train_step(x, y)

                if step % 10 == 0 or step == 1:
                    log_text = " ".join(f"{key}={value:.6f}" for key, value in logs.items())
                    print(f"step={step} {log_text}", flush=True)

                if step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(step)

                if step >= self.config.max_steps:
                    break

    def save_checkpoint(self, step: int) -> None:
        checkpoint = {
            "step": step,
            "config": self.config.__dict__,
        }
        checkpoint.update(self.method.checkpoint_state())
        path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save(checkpoint, path)

        # This file has the same format as the current inference weights.
        torch.save(self.method.inference_state(), self.output_dir / f"generator_step_{step:06d}.pt")
        print(f"saved checkpoint: {path}", flush=True)
