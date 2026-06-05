"""Dataset utilities for paired ERA5/RADKLIM training patches."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_file_list(path_or_dir: str | Path, pattern: str) -> list[Path]:
    path = Path(path_or_dir)
    if path.is_dir():
        files = sorted(path.glob(pattern))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No NetCDF files found: {path}")
    return files


def _sample_dim(ds) -> str | None:
    for name in ("sample", "sample_X", "sample_Y", "samples"):
        if name in ds.dims:
            return name
    return None


def _detect_variable(ds, preferred: str | None, candidates: tuple[str, ...]) -> str:
    if preferred is not None:
        if preferred not in ds:
            raise ValueError(f"Variable '{preferred}' not found")
        return preferred

    for name in candidates:
        if name in ds:
            return name

    raise ValueError(f"None of the candidate variables were found: {candidates}")


def _center_crop_tensor(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Center-crop a ``(time, height, width)`` tensor to target shape."""

    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor before channel insertion, got {tuple(tensor.shape)}")

    slices = []
    for current, target in zip(tensor.shape, target_shape):
        if current < target:
            raise ValueError(f"Cannot crop dimension from {current} to larger target {target}")
        start = (current - target) // 2
        slices.append(slice(start, start + target))

    return tensor[tuple(slices)]


def _apply_spatial_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对输入和目标做同一种随机旋转或翻转。"""

    op = rng.choice(("rot90", "rot270", "flip_h", "flip_v"))

    if op == "rot90":
        return torch.rot90(x, 1, dims=(-2, -1)), torch.rot90(y, 1, dims=(-2, -1))
    if op == "rot270":
        return torch.rot90(x, 3, dims=(-2, -1)), torch.rot90(y, 3, dims=(-2, -1))
    if op == "flip_h":
        return torch.flip(x, dims=(-1,)), torch.flip(y, dims=(-1,))
    return torch.flip(x, dims=(-2,)), torch.flip(y, dims=(-2,))


class PairedPatchDataset(Dataset):
    """Load paired low-resolution ERA5 patches and high-resolution targets.

    Expected model shapes after loading:
        x: ``(2, 16, 28, 28)``
        y: ``(1, 48, 168, 168)``

    The loader accepts either one NetCDF file or a directory of NetCDF files for
    each side. Large NetCDF files should normally be kept lazy so each training
    step loads only the selected sample.
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        x_pattern: str = "x*.nc",
        y_pattern: str = "y*.nc",
        x_variables: tuple[str, str] = ("cp", "lsp"),
        x_target_shape: tuple[int, int, int] = (16, 28, 28),
        y_variable: str | None = None,
        y_target_shape: tuple[int, int, int] = (48, 168, 168),
        augment_every_other: bool = True,
        load_into_memory: bool = False,
        seed: int = 42,
    ) -> None:
        import xarray as xr

        self.x_files = _as_file_list(x_path, x_pattern)[:2]
        self.y_files = _as_file_list(y_path, y_pattern)[:2]
        if len(self.x_files) != len(self.y_files):
            raise ValueError(
                "x_path 和 y_path 必须包含相同数量的匹配文件: "
                f"x={len(self.x_files)} using {x_pattern}, y={len(self.y_files)} using {y_pattern}"
            )

        self.x_variables = x_variables
        self.x_target_shape = x_target_shape
        self.y_variable = y_variable
        self.y_target_shape = y_target_shape
        self.augment_every_other = augment_every_other
        self.rng = random.Random(seed)

        self.x_datasets = [xr.open_dataset(path) for path in self.x_files]
        self.y_datasets = [xr.open_dataset(path) for path in self.y_files]
        if load_into_memory:
            self.x_datasets = [ds.load() for ds in self.x_datasets]
            self.y_datasets = [ds.load() for ds in self.y_datasets]

        self.index: list[tuple[int, int | None]] = []
        for file_idx, x_ds in enumerate(self.x_datasets):
            y_ds = self.y_datasets[file_idx]
            x_sample_dim = _sample_dim(x_ds)
            y_sample_dim = _sample_dim(y_ds)

            if (x_sample_dim is None) != (y_sample_dim is None):
                raise ValueError(
                    f"Sample dimension mismatch in pair {file_idx}: "
                    f"x={x_sample_dim}, y={y_sample_dim}"
                )

            if x_sample_dim is None:
                self.index.append((file_idx, None))
            else:
                n_x = x_ds.sizes[x_sample_dim]
                n_y = y_ds.sizes[y_sample_dim]
                if n_x != n_y:
                    raise ValueError(f"Sample count mismatch in pair {file_idx}: x={n_x}, y={n_y}")
                self.index.extend((file_idx, sample_idx) for sample_idx in range(n_x))

    def __len__(self) -> int:
        return len(self.index)

    def _select_sample(self, ds, sample_idx: int | None):
        sample_dim = _sample_dim(ds)
        if sample_dim is None:
            return ds
        if sample_idx is None:
            raise ValueError("sample_idx is required for sampled datasets")
        return ds.isel({sample_dim: sample_idx})

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, sample_idx = self.index[idx]
        x_ds = self._select_sample(self.x_datasets[file_idx], sample_idx)
        y_ds = self._select_sample(self.y_datasets[file_idx], sample_idx)

        x_arrays = []
        for name in self.x_variables:
            if name not in x_ds:
                raise ValueError(f"Input variable '{name}' not found")
            x_array = torch.from_numpy(np.asarray(x_ds[name].values, dtype=np.float32))
            x_array = _center_crop_tensor(x_array, self.x_target_shape)
            x_arrays.append(x_array)
        x = torch.stack(x_arrays, dim=0)

        y_variable = _detect_variable(
            y_ds,
            self.y_variable,
            candidates=("rainfall_amount", "sample_high_res"),
        )
        y = torch.from_numpy(np.asarray(y_ds[y_variable].values, dtype=np.float32))
        if y.ndim == 3:
            y = _center_crop_tensor(y, self.y_target_shape)
            y = y.unsqueeze(0)
        elif y.ndim == 4 and y.shape[0] == 1:
            y = _center_crop_tensor(y[0], self.y_target_shape).unsqueeze(0)
        else:
            raise ValueError(f"Expected y shape (T, H, W) or (1, T, H, W), got {tuple(y.shape)}")

        if x.shape != (2, 16, 28, 28):
            raise ValueError(f"Expected x shape (2, 16, 28, 28), got {tuple(x.shape)}")
        if y.shape != (1, 48, 168, 168):
            raise ValueError(f"Expected y shape (1, 48, 168, 168), got {tuple(y.shape)}")

        if torch.isnan(x).any() or torch.isnan(y).any():
            raise ValueError(f"NaN found in sample {idx}")

        if self.augment_every_other and idx % 2 == 1:
            x, y = _apply_spatial_augmentation(x, y, self.rng)

        return x, y
