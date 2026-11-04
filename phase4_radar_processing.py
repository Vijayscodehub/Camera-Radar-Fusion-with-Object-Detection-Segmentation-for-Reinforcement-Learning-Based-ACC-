import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
SAMPLE_INDEX = 95
MIN_DISTANCE_M = 2.0
MAX_DISTANCE_M = 100.0
INVALID_STATE_ROW = 14


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    output_directory = Path("outputs/radar").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    radar_data = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    radar_path = dataset_root / radar_data["filename"]
    radar = RadarPointCloud.from_file(str(radar_path))

    x_m = radar.points[0]
    y_m = radar.points[1]
    vx_comp_mps = radar.points[8]
    vy_comp_mps = radar.points[9]
    invalid_state = radar.points[INVALID_STATE_ROW]
    distance_m = np.hypot(x_m, y_m)

    valid_targets = (
        (x_m > 0)
        & (distance_m >= MIN_DISTANCE_M)
        & (distance_m <= MAX_DISTANCE_M)
        & (invalid_state == 0)
    )
    target_indices = np.flatnonzero(valid_targets)

    targets = [
        {
            "radar_point_index": int(index),
            "x_m": round(float(x_m[index]), 2),
            "y_m": round(float(y_m[index]), 2),
            "distance_m": round(float(distance_m[index]), 2),
            "relative_velocity_mps": round(float(vx_comp_mps[index]), 2),
            "lateral_velocity_mps": round(float(vy_comp_mps[index]), 2),
        }
        for index in target_indices
    ]

    output_stem = f"radar_front_sample_{sample_index:03d}"
    targets_path = output_directory / f"{output_stem}_targets.json"
    visualization_path = output_directory / f"{output_stem}_top_down.png"
    targets_path.write_text(json.dumps(targets, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(10, 7))
    points = axis.scatter(
        x_m[valid_targets],
        y_m[valid_targets],
        c=vx_comp_mps[valid_targets],
        cmap="coolwarm",
        s=50,
        edgecolors="black",
        linewidths=0.4,
    )
    axis.scatter(0, 0, c="black", marker="^", s=120, label="Ego vehicle")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)
    axis.set_title("Filtered RADAR_FRONT Targets")
    axis.set_xlabel("Forward distance from radar (m)")
    axis.set_ylabel("Left distance from radar (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.colorbar(points, ax=axis, label="Compensated forward velocity (m/s)")
    figure.tight_layout()
    figure.savefig(visualization_path, dpi=150)
    plt.close(figure)

    print(f"Input radar file: {radar_data['filename']}")
    print(f"Sample index: {sample_index}")
    print(f"Raw radar points: {radar.points.shape[1]}")
    print(f"Filtered radar targets: {len(targets)}")
    print(f"Saved targets JSON: {targets_path}")
    print(f"Saved top-down visualization: {visualization_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract filtered RADAR_FRONT targets from a nuScenes sample."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    )
    arguments = parser.parse_args()
    main(arguments.sample_index)