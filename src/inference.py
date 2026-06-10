from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from model import GRUSOCModel


def _safe_torch_load(model_path: Path, device: torch.device) -> dict:
    """
    Wczytuje checkpoint modelu.
    Obsługuje różne wersje PyTorch.
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_trained_model(model_path: Path, device: torch.device | None = None) -> tuple[GRUSOCModel, dict]:
    """
    Wczytuje wytrenowany model GRU oraz checkpoint z parametrami normalizacji.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _safe_torch_load(model_path, device)

    model = GRUSOCModel(
        input_size=5,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def normalize_window(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalizuje okno wejściowe tak samo jak podczas treningu.
    """
    return ((window - mean) / std).astype(np.float32)


def predict_soc_from_window(
    model: GRUSOCModel,
    window_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> float:
    """
    Wykonuje predykcję SOC dla jednego okna czasowego.

    window_raw shape:
        [seq_len, 5]

    Zwraca:
        SOC w zakresie [0, 1]
    """
    window_norm = normalize_window(window_raw, mean, std)

    x = torch.from_numpy(window_norm).unsqueeze(0).to(device)
    # shape: [1, seq_len, 5]

    with torch.no_grad():
        pred = model(x).item()

    return float(pred)


class RealtimeSOCBuffer:
    """
    Bufor do predykcji w czasie rzeczywistym.

    Do bufora dodawane są kolejne próbki:
        [voltage, current, temperature, avg_voltage, avg_current]

    Gdy bufor ma długość seq_len, można wykonać predykcję SOC.
    """

    def __init__(
        self,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        model: GRUSOCModel,
        device: torch.device,
    ) -> None:
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        self.model = model
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=seq_len)

    def add_sample(self, sample: np.ndarray) -> None:
        """
        Dodaje jedną próbkę do bufora.
        sample shape: [5]
        """
        sample = np.asarray(sample, dtype=np.float32)

        if sample.shape != (5,):
            raise ValueError(f"Expected sample shape (5,), got {sample.shape}")

        self.buffer.append(sample)

    def is_ready(self) -> bool:
        """
        Sprawdza, czy bufor ma już pełne okno.
        """
        return len(self.buffer) == self.seq_len

    def predict(self) -> float:
        """
        Wykonuje predykcję SOC, jeśli bufor jest pełny.
        """
        if not self.is_ready():
            raise RuntimeError("Buffer is not full yet. Cannot predict SOC.")

        window = np.stack(list(self.buffer), axis=0)
        return predict_soc_from_window(
            model=self.model,
            window_raw=window,
            mean=self.mean,
            std=self.std,
            device=self.device,
        )