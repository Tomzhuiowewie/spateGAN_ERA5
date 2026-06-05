#!/usr/bin/env python3
"""Standalone training entry point for the reproduction folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reproduced spateGAN-ERA5 cGAN")
    parser.add_argument("--config", type=Path, default=Path("training_reproduction/config_train.yml"))
    parser.add_argument("--x-path", type=str, help="Override training input X path")
    parser.add_argument("--y-path", type=str, help="Override training target Y path")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Override device")
    return parser.parse_args()


def load_config(path: Path) -> TrainingConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TrainingConfig(**raw)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.x_path:
        config.x_path = args.x_path
    if args.y_path:
        config.y_path = args.y_path
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.device:
        config.device = args.device

    trainer = Trainer(config)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
