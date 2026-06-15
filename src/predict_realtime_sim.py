from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

#import numpy as np
import torch

from data_utils import list_mat_files, load_mat_file, resolve_default_data_root
from inference import RealtimeSOCBuffer, load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate real-time SOC prediction using a trained GRU model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained .pt model. Default: models/soc_gru_model.pt",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to dataset root. Default: data/LGHG2@n10C_to_25degC",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional path to one .mat file. If not provided, first Test file is used.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="How many samples to simulate. Use 0 for full sequence.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between samples in seconds. Example: 1.0 simulates 1 Hz.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="Print prediction every N samples.",
    )

    return parser.parse_args()


def get_default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models" / "soc_gru_model.pt"


def get_outputs_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "realtime_sim"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "realtime_sim_predictions.csv"


def choose_test_file(data_root: Path, test_file_arg: str | None) -> Path:
    if test_file_arg:
        test_file = Path(test_file_arg)
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        return test_file

    test_dir = data_root / "Test"
    test_files = list_mat_files(test_dir)

    if not test_files:
        raise FileNotFoundError(f"No .mat test files found in {test_dir}")

    return test_files[0]


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path) if args.model_path else get_default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "First train the model, for example:\n"
            "python src/main.py --stride 32 --epochs 10"
        )

    data_root = Path(args.data_root) if args.data_root else resolve_default_data_root()
    test_file = choose_test_file(data_root, args.test_file)

    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    print(f"Test file: {test_file}")

    model, checkpoint = load_trained_model(model_path, device)

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    seq_len = int(checkpoint["seq_len"])

    x_raw, y_true = load_mat_file(test_file)

    if args.max_samples and args.max_samples > 0:
        x_raw = x_raw[: args.max_samples]
        y_true = y_true[: args.max_samples]

    soc_buffer = RealtimeSOCBuffer(
        seq_len=seq_len,
        mean=mean,
        std=std,
        model=model,
        device=device,
    )

    output_csv = get_outputs_path()

    print(f"Sequence length required by model: {seq_len}")
    print(f"Samples to simulate: {len(x_raw)}")
    print(f"Saving results to: {output_csv}")
    print()
    print("Starting realtime simulation...")

    rows = []

    for idx, sample in enumerate(x_raw):
        # sample ma już 5 cech:
        # voltage, current, temperature, average voltage, average current
        soc_buffer.add_sample(sample)

        if soc_buffer.is_ready():
            y_pred = soc_buffer.predict()
            y_ref = float(y_true[idx])
            error_percent = (y_pred - y_ref) * 100.0

            rows.append(
                {
                    "sample_index": idx,
                    "soc_true": y_ref,
                    "soc_pred": y_pred,
                    "soc_true_percent": y_ref * 100.0,
                    "soc_pred_percent": y_pred * 100.0,
                    "error_percent_points": error_percent,
                }
            )

            if idx % args.print_every == 0:
                print(
                    f"sample={idx} | "
                    f"SOC true={y_ref * 100.0:.2f}% | "
                    f"SOC pred={y_pred * 100.0:.2f}% | "
                    f"error={error_percent:.2f} p.p."
                )

        if args.delay > 0:
            time.sleep(args.delay)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "soc_true",
                "soc_pred",
                "soc_true_percent",
                "soc_pred_percent",
                "error_percent_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Realtime simulation finished.")
    print(f"Saved CSV to: {output_csv}")


if __name__ == "__main__":
    main()