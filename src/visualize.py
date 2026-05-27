from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _prepare_plot_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int | None = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Usuwa wartości NaN z predykcji i opcjonalnie ogranicza liczbę punktów,
    żeby wykresy nie były zbyt ciężkie.
    """
    valid_mask = ~np.isnan(y_pred)

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    time_idx = np.arange(len(y_true))[valid_mask]

    if max_points is not None and len(time_idx) > max_points:
        step = max(len(time_idx) // max_points, 1)
        time_idx = time_idx[::step]
        y_true_valid = y_true_valid[::step]
        y_pred_valid = y_pred_valid[::step]

    return time_idx, y_true_valid, y_pred_valid


def plot_soc_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
    max_points: int | None = 5000,
) -> None:
    """
    Tworzy wykres SOC rzeczywisty vs SOC przewidywany.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    time_idx, y_true_valid, y_pred_valid = _prepare_plot_data(
        y_true,
        y_pred,
        max_points=max_points,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(time_idx, y_true_valid * 100.0, label="SOC rzeczywisty")
    plt.plot(time_idx, y_pred_valid * 100.0, label="SOC przewidywany")
    plt.title(title)
    plt.xlabel("Próbka czasowa")
    plt.ylabel("SOC [%]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
    max_points: int | None = 5000,
) -> None:
    """
    Tworzy wykres błędu predykcji w czasie.
    Błąd = SOC_pred - SOC_true.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    time_idx, y_true_valid, y_pred_valid = _prepare_plot_data(
        y_true,
        y_pred,
        max_points=max_points,
    )

    error = (y_pred_valid - y_true_valid) * 100.0

    plt.figure(figsize=(12, 5))
    plt.plot(time_idx, error, label="Błąd predykcji")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Próbka czasowa")
    plt.ylabel("Błąd SOC [p.p.]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()