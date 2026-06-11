#!/usr/bin/env python3
"""Run tiny CPU comparison experiments for the implemented training methods."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny CPU comparison experiments")
    parser.add_argument("--output-root", type=Path, default=Path("training_reproduction/checkpoints/cpu_comparison"))
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--validation-max-batches", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.01)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=2)
    return parser.parse_args()


def make_config(name: str, method: str, args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig(
        x_path="data/training_data",
        y_path="data/training_data",
        x_pattern="x_train*.nc",
        y_pattern="y_train*.nc",
        y_variable="sample_high_res",
        training_method=method,
        output_dir=str(args.output_root / name),
        batch_size=1,
        max_steps=args.max_steps,
        checkpoint_interval=999,
        validation_x_path="data/validation_data",
        validation_y_path="data/validation_data",
        validation_x_pattern="x_val*.nc",
        validation_y_pattern="y_val*.nc",
        validation_interval=args.validation_interval,
        validation_batch_size=1,
        validation_max_batches=args.validation_max_batches,
        save_visualizations=False,
        device="cpu",
        seed=42,
        num_workers=0,
        load_into_memory=False,
        dataset_cache_size=1,
        train_fraction=args.train_fraction,
        subset_seed=42,
        generator_lr=1e-4,
        ensemble_size=1,
        base_channels=args.base_channels,
        diffusion_steps=args.diffusion_steps,
        sampling_steps=args.sampling_steps,
        pde_weight=0.01,
        mass_weight=0.05,
        smooth_weight=0.005,
    )
    if name == "erd_no_physics":
        config.pde_weight = 0.0
        config.mass_weight = 0.0
        config.smooth_weight = 0.0
    if method.startswith("era5_"):
        config.max_steps = 1
        config.validation_interval = 1
    return config


def read_last_metrics(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else {}


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("era5_nearest", "era5_nearest"),
        ("era5_trilinear", "era5_trilinear"),
        ("cnn_full_1pct", "cnn"),
        ("st_resunet_full_1pct", "st_resunet"),
        ("erd_no_physics", "pg_erd"),
        ("pg_erd_full", "pg_erd"),
    ]

    summary_rows = []
    for name, method in experiments:
        print(f"\n=== running {name} ({method}) ===", flush=True)
        config = make_config(name, method, args)
        trainer = Trainer(config)
        trainer.train()
        validation_metrics = read_last_metrics(Path(config.output_dir) / "validation_metrics.csv")
        summary_rows.append({"experiment": name, "method": method, **validation_metrics})

    summary_path = output_root / "summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row})
    preferred = [
        "experiment",
        "method",
        "step",
        "samples",
        "val_mae",
        "val_mse",
        "val_rmse",
        "val_bias",
        "val_wet_mae",
        "val_crps",
        "val_pde",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nsummary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
