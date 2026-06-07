"""Training loop for the paper reproduction and compatible experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    validation_x_path: str = "data/validation_data"
    validation_y_path: str = "data/validation_data"
    validation_x_pattern: str = "x_val*.nc"
    validation_y_pattern: str = "y_val*.nc"
    validation_interval: int | None = None
    validation_interval_epochs: int = 1
    validation_batch_size: int = 2
    validation_max_batches: int | None = None
    save_visualizations: bool = True
    visualization_vmin: float = 0.01
    visualization_vmax: float = 20.0
    visualization_sample_index: int = 0
    visualization_timestep: int = 12
    visualization_seed: int = 4
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    load_into_memory: bool = False
    dataset_cache_size: int = 16
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
            dataset_cache_size=config.dataset_cache_size,
            seed=config.seed,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )

        self.validation_dataset = PairedPatchDataset(
            config.validation_x_path,
            config.validation_y_path,
            x_pattern=config.validation_x_pattern,
            y_pattern=config.validation_y_pattern,
            x_target_shape=config.x_target_shape,
            y_variable=config.y_variable,
            y_target_shape=config.y_target_shape,
            augment_every_other=False,
            load_into_memory=config.load_into_memory,
            dataset_cache_size=config.dataset_cache_size,
            seed=config.seed,
        )
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.validation_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )

        self.method = build_training_method(config.training_method, self.device, config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir = self.output_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics_path = self.output_dir / "train_metrics.csv"
        self.validation_metrics_path = self.output_dir / "validation_metrics.csv"
        self.best_val_mae = float("inf")
        if config.validation_interval is not None:
            self.validation_interval = config.validation_interval
        else:
            self.validation_interval = len(self.loader) * config.validation_interval_epochs
        print(
            f"train_samples={len(self.dataset)} steps_per_epoch={len(self.loader)} "
            f"validation_interval={self.validation_interval}",
            flush=True,
        )

    def train(self) -> None:
        step = 0
        progress = tqdm(
            total=self.config.max_steps,
            desc="training",
            unit="step",
            dynamic_ncols=True,
        )
        try:
            while step < self.config.max_steps:
                for epoch_step, (x, y) in enumerate(self.loader, start=1):
                    step += 1
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    logs = self.method.train_step(x, y)
                    epoch = (step - 1) // len(self.loader) + 1
                    self._append_metrics(
                        self.train_metrics_path,
                        {"step": step, "epoch": epoch, **logs},
                    )
                    progress.set_postfix(
                        epoch=f"{epoch}:{epoch_step}/{len(self.loader)}",
                        **{key: f"{value:.4f}" for key, value in logs.items()},
                    )
                    progress.update(1)

                    if step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(step)

                    if (
                        self.validation_interval > 0
                        and (
                            step % self.validation_interval == 0
                            or step == self.config.max_steps
                        )
                    ):
                        metrics = self.validate(step)
                        progress.set_postfix(
                            epoch=f"{epoch}:{epoch_step}/{len(self.loader)}",
                            **{key: f"{value:.4f}" for key, value in logs.items()},
                            **{key: f"{value:.4f}" for key, value in metrics.items()},
                        )

                    if step >= self.config.max_steps:
                        break
        finally:
            progress.close()

    def validate(self, step: int) -> dict[str, float]:
        totals: dict[str, float] = {}
        sample_count = 0
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.config.seed)
            validation_batches = len(self.validation_loader)
            validation_iter = iter(self.validation_loader)
            if self.config.validation_max_batches is not None:
                validation_batches = min(
                    validation_batches,
                    self.config.validation_max_batches,
                )
                validation_iter = islice(validation_iter, validation_batches)
            validation_progress = tqdm(
                validation_iter,
                total=validation_batches,
                desc=f"validation step={step}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch_index, (x, y) in enumerate(validation_progress):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logs, _ = self.method.validation_step(
                    x,
                    y,
                    seed=self.config.seed + batch_index * self.config.ensemble_size,
                )
                batch_size = x.shape[0]
                sample_count += batch_size
                for key, value in logs.items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size

        if sample_count == 0:
            raise ValueError("Validation loader produced no samples")

        metrics = {key: value / sample_count for key, value in totals.items()}
        epoch = step / len(self.loader)
        self._append_metrics(
            self.validation_metrics_path,
            {"step": step, "epoch": epoch, "samples": sample_count, **metrics},
        )
        if self.config.save_visualizations:
            visualization_sample = self._notebook_visualization_sample()
            self._save_validation_visualization(step, *visualization_sample)
            self._save_downscaling_sequence(step, *visualization_sample)
            self._save_metric_curves()
        log_text = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        tqdm.write(f"validation step={step} samples={sample_count} {log_text}")

        if metrics["val_mae"] < self.best_val_mae:
            self.best_val_mae = metrics["val_mae"]
            torch.save(
                {
                    "step": step,
                    "val_metrics": metrics,
                    "config": self.config.__dict__,
                    **self.method.checkpoint_state(),
                },
                self.output_dir / "best_checkpoint.pt",
            )
            torch.save(
                self.method.inference_state(),
                self.output_dir / "best_generator.pt",
            )
            tqdm.write(f"saved best model: step={step} val_mae={self.best_val_mae:.6f}")

        return metrics

    def _notebook_visualization_sample(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_index = self.config.visualization_sample_index
        if not 0 <= sample_index < len(self.validation_dataset):
            raise ValueError(
                f"visualization_sample_index must be between 0 and "
                f"{len(self.validation_dataset) - 1}, got {sample_index}"
            )

        x, target = self.validation_dataset[sample_index]
        x_batch = x.unsqueeze(0).to(self.device)
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            prediction = self.method.visualization_prediction(
                x_batch,
                seed=self.config.visualization_seed,
            )[0]
        return x, target, prediction.detach().float().cpu()

    @staticmethod
    def _append_metrics(path: Path, values: dict[str, float | int]) -> None:
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(values))
            if write_header:
                writer.writeheader()
            writer.writerow(values)

    def _save_validation_visualization(
        self,
        step: int,
        x: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        input_total = x.sum(dim=(0, 1)).numpy()
        target_total = target.sum(dim=(0, 1)).numpy()
        prediction_total = prediction.sum(dim=(0, 1)).numpy()
        error = abs(prediction_total - target_total)
        rain_vmax = max(float(target_total.max()), float(prediction_total.max()), 1e-8)

        figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
        panels = (
            ("ERA5 input total", input_total, "Blues", None),
            ("Observed total", target_total, "Blues", rain_vmax),
            ("Predicted total", prediction_total, "Blues", rain_vmax),
            ("Absolute error", error, "magma", None),
        )
        for axis, (title, image, cmap, vmax) in zip(axes, panels):
            plot = axis.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)

        figure.suptitle(f"Validation sample at step {step}")
        figure.savefig(
            self.visualization_dir / f"validation_step_{step:06d}.png",
            dpi=160,
        )
        plt.close(figure)

    def _save_downscaling_sequence(
        self,
        step: int,
        x: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        # Match the notebook inference view: crop model borders and convert the
        # 10-minute target/prediction amounts to the displayed mm/h rate.
        era5 = x.sum(dim=0)[5:-5, 8:-8, 8:-8].numpy()
        target_sequence = target[0, 6:-6, 12:-12, 12:-12].numpy() * 6
        prediction_sequence = prediction[0, 6:-6, 12:-12, 12:-12].numpy() * 6

        sequence_start = self.config.visualization_timestep
        if sequence_start % 6 != 0:
            raise ValueError("visualization_timestep must be a multiple of 6")
        if not 0 <= sequence_start <= target_sequence.shape[0] - 6:
            raise ValueError(
                f"visualization_timestep must be between 0 and "
                f"{target_sequence.shape[0] - 6}, got {sequence_start}"
            )
        era5_index = min(sequence_start // 6, era5.shape[0] - 1)

        figure = plt.figure(figsize=(14, 7))
        grid = gridspec.GridSpec(
            3,
            7,
            figure=figure,
            width_ratios=[1] * 6 + [0.05],
        )

        axis = figure.add_subplot(grid[0, 0])
        image = axis.imshow(
            era5[era5_index],
            cmap="turbo",
            vmin=self.config.visualization_vmin,
            vmax=self.config.visualization_vmax,
        )
        axis.set_title("ERA5 TP, t")
        axis.axis("off")

        for offset in range(6):
            target_axis = figure.add_subplot(grid[1, offset])
            image = target_axis.imshow(
                target_sequence[sequence_start + offset],
                cmap="turbo",
                vmin=self.config.visualization_vmin,
                vmax=self.config.visualization_vmax,
            )
            prefix = "RADKLIM-YW " if offset == 0 else ""
            target_axis.set_title(f"{prefix}t+{offset * 10}min.")
            target_axis.axis("off")

            prediction_axis = figure.add_subplot(grid[2, offset])
            image = prediction_axis.imshow(
                prediction_sequence[sequence_start + offset],
                cmap="turbo",
                vmin=self.config.visualization_vmin,
                vmax=self.config.visualization_vmax,
            )
            prefix = "spateGAN-ERA5 " if offset == 0 else ""
            prediction_axis.set_title(f"{prefix}t+{offset * 10}min.")
            prediction_axis.axis("off")

        colorbar_axis = figure.add_subplot(grid[-1, 6])
        figure.colorbar(image, cax=colorbar_axis, label="Rain [mm/h]")
        figure.suptitle(
            f"Validation downscaling sequence at step {step} | "
            f"sample={self.config.visualization_sample_index} "
            f"timestep={self.config.visualization_timestep} "
            f"seed={self.config.visualization_seed}"
        )
        figure.tight_layout()
        figure.savefig(
            self.visualization_dir / f"downscaling_sequence_step_{step:06d}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _save_metric_curves(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_rows = self._read_metrics(self.train_metrics_path)
        validation_rows = self._read_metrics(self.validation_metrics_path)
        if not train_rows:
            return

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        train_keys = [key for key in ("d_loss", "g_loss", "g_adv", "g_l1", "loss") if key in train_rows[0]]
        for key in train_keys:
            axes[0].plot(
                [float(row["step"]) for row in train_rows],
                [float(row[key]) for row in train_rows],
                label=key,
                linewidth=1,
            )
        axes[0].set_title("Training metrics")
        axes[0].set_xlabel("Step")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        for key in ("val_mae", "val_mse"):
            if validation_rows and key in validation_rows[0]:
                axes[1].plot(
                    [float(row["step"]) for row in validation_rows],
                    [float(row[key]) for row in validation_rows],
                    marker="o",
                    label=key,
                )
        axes[1].set_title("Validation metrics")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        figure.savefig(self.visualization_dir / "metric_curves.png", dpi=160)
        plt.close(figure)

    @staticmethod
    def _read_metrics(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

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
        tqdm.write(f"saved checkpoint: {path}")
