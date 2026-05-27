from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

#sciezka
def resolve_default_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "data" / "LGHG2@n10C_to_25degC"

#load train/test files
def load_mat_file(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(mat_path)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"File {mat_path} does not contain both 'X' and 'Y' variables.")

    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["Y"], dtype=np.float32)

    # Dataset is usually stored as [features, time]
    if x.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {x.shape} in {mat_path}")

    if x.shape[0] == 5:
        x = x.T  # -> [time, features]
    elif x.shape[1] != 5:
        raise ValueError(f"Expected 5 features in X, got shape {x.shape} in {mat_path}")

    y = y.reshape(-1)
    if len(y) != len(x):
        raise ValueError(f"Length mismatch: X has {len(x)} time steps, Y has {len(y)} in {mat_path}")

    return x, y


def list_mat_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.mat"))


def compute_feature_stats(feature_arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(list(feature_arrays))
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


class WindowedSOCDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_len: int = 256, stride: int = 1) -> None:
        if len(x) != len(y):
            raise ValueError("X and Y must have the same number of time steps.")
        if len(x) < seq_len:
            raise ValueError(f"Sequence length {len(x)} is shorter than seq_len={seq_len}")
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.indices = list(range(seq_len - 1, len(x), stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1
        x_window = self.x[start_idx : end_idx + 1]
        y_target = self.y[end_idx]
        return torch.from_numpy(x_window), torch.tensor(y_target, dtype=torch.float32)


@dataclass
class SequenceMetrics:
    rmse_percent: float
    mae_percent: float
    max_percent: float


def compute_sequence_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SequenceMetrics:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)) * 100.0)
    mae = float(np.mean(np.abs(err)) * 100.0)
    max_abs = float(np.max(np.abs(err)) * 100.0)
    return SequenceMetrics(rmse_percent=rmse, mae_percent=mae, max_percent=max_abs)
