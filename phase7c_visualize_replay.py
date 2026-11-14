import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ACTION_COLORS = {
    "BRAKE": "tab:red",
    "MAINTAIN": "tab:orange",
    "ACCELERATE": "tab:green",
}


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def main(results_path: Path) -> None:
    if not results_path.is_file():
        raise FileNotFoundError(f"Replay results CSV does not exist: {results_path}")

    with results_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("Replay results CSV has no data rows.")

    frame_numbers = [int(row["frame_number"]) for row in rows]
    distances_m = [optional_float(row["lead_distance_m"]) for row in rows]
    velocities_mps = [optional_float(row["lead_relative_speed_mps"]) for row in rows]
    ttc_s = [optional_float(row["ttc_s"]) for row in rows]
    actions = [row["recommended_action"] for row in rows]

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 9),
        sharex=True,
        layout="constrained",
    )

    axes[0].plot(frame_numbers, distances_m, marker="o", color="tab:blue")
    axes[0].set_ylabel("Raw lead distance (m)")
    axes[0].set_title("Camera-Radar Replay: Raw Phase 6 Lead Measurements")
    axes[0].grid(alpha=0.3)

    axes[1].plot(frame_numbers, velocities_mps, marker="o", color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Radar velocity (m/s)")
    axes[1].grid(alpha=0.3)

    finite_ttc = [value if value is not None else float("nan") for value in ttc_s]
    axes[2].plot(frame_numbers, finite_ttc, marker="o", color="tab:gray", label="TTC")
    for frame_number, action in zip(frame_numbers, actions):
        axes[2].scatter(
            frame_number,
            0,
            color=ACTION_COLORS[action],
            s=65,
            zorder=3,
        )
    for action, color in ACTION_COLORS.items():
        axes[2].scatter([], [], color=color, label=action)
    axes[2].set_xlabel("Replay frame number")
    axes[2].set_ylabel("TTC (s)")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.3)
    axes[2].legend(title="Action markers", ncol=4)

    output_path = results_path.with_name(f"{results_path.stem}_plot.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Replay frames plotted: {len(rows)}")
    print(f"Saved replay plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot raw lead-state and rule-decision results from a replay CSV."
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to a replay results CSV created by phase7b_run_replay_pipeline.py.",
    )
    arguments = parser.parse_args()
    main(arguments.results_path) #It reads the results table, draws 3 graphs (distance, speed, time-to-collision) with colored dots for brake, maintain, or accelerate, and saves them as one PNG image.