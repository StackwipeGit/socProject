from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import csv
from torch import nn
from torch.utils.data import DataLoader

from data_utils import (
    WindowedSOCDataset,
    compute_feature_stats,
    compute_sequence_metrics,
    list_mat_files,
    load_mat_file,
    normalize_features,
    resolve_default_data_root,
)
from model import GRUSOCModel
from visualize import plot_soc_prediction, plot_prediction_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU model for battery SOC estimation.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root.")
    parser.add_argument("--seq_len", type=int, default=256, help="Sliding window length.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride for training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=64, help="GRU hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="GRU dropout.")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save .pt model.")
    return parser.parse_args()


def evaluate_on_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yp = model(xb)
            loss = criterion(yp, yb)
            losses.append(loss.item())
            preds.append(yp.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return float(np.mean(losses)), rmse, mae


def predict_full_sequence(model: nn.Module, x_seq: np.ndarray, seq_len: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = np.full(len(x_seq), np.nan, dtype=np.float32)
    with torch.no_grad():
        for end_idx in range(seq_len - 1, len(x_seq)):
            start_idx = end_idx - seq_len + 1
            x_window = torch.from_numpy(x_seq[start_idx : end_idx + 1]).unsqueeze(0).to(device)
            pred = model(x_window).item()
            preds[end_idx] = pred
    return preds


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else resolve_default_data_root()

    train_dir = data_root / "Train"
    test_dir = data_root / "Test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at:\n  {train_dir}\n  {test_dir}\n"
            "Place the dataset under data/LGHG2@n10C_to_25degC or use --data_root."
        )

    train_files = list_mat_files(train_dir)
    test_files = list_mat_files(test_dir)

    if not train_files:
        raise FileNotFoundError(f"No .mat files found in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No .mat files found in {test_dir}")

    x_train_raw, y_train = load_mat_file(train_files[0])
    mean, std = compute_feature_stats([x_train_raw])
    x_train = normalize_features(x_train_raw, mean, std)

    train_ds = WindowedSOCDataset(x_train, y_train, seq_len=args.seq_len, stride=args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    x_val_raw, y_val = load_mat_file(test_files[0])
    x_val = normalize_features(x_val_raw, mean, std)
    val_ds = WindowedSOCDataset(x_val, y_val, seq_len=args.seq_len, stride=max(args.seq_len // 4, 1))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUSOCModel(
        input_size=5,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    print(f"Dataset root: {data_root}")
    print(f"Train file: {train_files[0].name}")
    print(f"Test files: {[p.name for p in test_files]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yp = model(xb)
            loss = criterion(yp, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_mse = float(np.mean(batch_losses))
        val_mse, val_rmse, val_mae = evaluate_on_loader(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val_mse:.6f} | "
            f"val_rmse={val_rmse:.6f} | "
            f"val_mae={val_mae:.6f}"
        )

    project_root = Path(__file__).resolve().parents[1]
    plots_dir = project_root / "outputs" / "plots"
    metrics_dir = project_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = metrics_dir / "test_metrics.csv"

    print("\nPer-test-file sequence metrics:")

    metrics_rows = []

    for test_file in test_files:
        x_test_raw, y_test = load_mat_file(test_file)
        x_test = normalize_features(x_test_raw, mean, std)
        y_pred = predict_full_sequence(model, x_test, args.seq_len, device)

        valid_mask = ~np.isnan(y_pred)
        metrics = compute_sequence_metrics(y_test[valid_mask], y_pred[valid_mask])

        print(
            f"{test_file.name} | "
            f"RMSE={metrics.rmse_percent:.3f}% | "
            f"MAE={metrics.mae_percent:.3f}% | "
            f"MAX={metrics.max_percent:.3f}%"
        )

        safe_name = test_file.stem.replace("@", "_").replace(" ", "_")

        plot_soc_prediction(
            y_true=y_test,
            y_pred=y_pred,
            title=f"SOC rzeczywisty vs przewidywany - {test_file.stem}",
            save_path=plots_dir / f"{safe_name}_soc_prediction.png",
        )

        plot_prediction_error(
            y_true=y_test,
            y_pred=y_pred,
            title=f"Błąd predykcji SOC - {test_file.stem}",
            save_path=plots_dir / f"{safe_name}_prediction_error.png",
        )

        metrics_rows.append(
            {
                "file": test_file.name,
                "rmse_percent": metrics.rmse_percent,
                "mae_percent": metrics.mae_percent,
                "max_percent": metrics.max_percent,
            }
        )

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "rmse_percent", "mae_percent", "max_percent"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\nSaved plots to: {plots_dir}")
    print(f"Saved metrics to: {metrics_csv_path}")


    default_model_path = Path(__file__).resolve().parents[1] / "models" / "soc_gru_model.pt"
    save_path = Path(args.save_model) if args.save_model else default_model_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        save_path,
    )
    print(f"\nSaved model to: {save_path}")


if __name__ == "__main__":
    main()
