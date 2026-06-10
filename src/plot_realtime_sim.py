from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot realtime SOC simulation results from CSV."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to realtime_sim_predictions.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output plot PNG",
    )

    return parser.parse_args()


def get_default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]

    csv_path = (
        project_root
        / "outputs"
        / "realtime_sim"
        / "realtime_sim_predictions.csv"
    )

    output_path = (
        project_root
        / "outputs"
        / "realtime_sim"
        / "realtime_sim_plot.png"
    )

    return csv_path, output_path


def main() -> None:
    args = parse_args()

    default_csv_path, default_output_path = get_default_paths()

    csv_path = Path(args.csv_path) if args.csv_path else default_csv_path
    output_path = Path(args.output_path) if args.output_path else default_output_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "First run:\n"
            "python src/predict_realtime_sim.py"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_columns = {
        "sample_index",
        "soc_true_percent",
        "soc_pred_percent",
        "error_percent_points",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV file is missing columns: {missing_columns}")

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["sample_index"],
        df["soc_true_percent"],
        label="SOC rzeczywisty",
    )
    plt.plot(
        df["sample_index"],
        df["soc_pred_percent"],
        label="SOC przewidywany",
    )

    plt.title("Symulacja predykcji SOC w czasie rzeczywistym")
    plt.xlabel("Numer próbki")
    plt.ylabel("SOC [%]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    error_output_path = output_path.with_name("realtime_sim_error_plot.png")

    plt.figure(figsize=(12, 5))
    plt.plot(
        df["sample_index"],
        df["error_percent_points"],
        label="Błąd predykcji",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Błąd predykcji SOC w symulacji realtime")
    plt.xlabel("Numer próbki")
    plt.ylabel("Błąd SOC [p.p.]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(error_output_path, dpi=150)
    plt.close()

    print(f"Saved SOC plot to: {output_path}")
    print(f"Saved error plot to: {error_output_path}")


if __name__ == "__main__":
    main()