from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
OUTPUT_PATH = Path("outputs/phase0/radar_front_top_down.png")


def main() -> None:
    dataset_root = DATA_ROOT.resolve()
    output_path = OUTPUT_PATH.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    sample = nusc.sample[0]
    radar_data = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    radar_path = dataset_root / radar_data["filename"]
    radar = RadarPointCloud.from_file(str(radar_path))

    forward_position_m = radar.points[0] #row 0 is how far forward each dot is
    left_position_m = radar.points[1] #row 1 is how far left each dot is
    compensated_forward_velocity_mps = radar.points[8] #row 8 is the compensated forward velocity

    figure, axis = plt.subplots(figsize=(10, 7))
    points = axis.scatter(
        forward_position_m,
        left_position_m,
        c=compensated_forward_velocity_mps,
        cmap="coolwarm",
        s=50,
        edgecolors="black",
        linewidths=0.4,
    )
    axis.scatter(0, 0, c="black", marker="^", s=120, label="Ego vehicle") #Puts one black triangle at the middle (0, 0) to stand for our car
    #horizontal and a vertical gray line through zero,
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)
    axis.set_title("RADAR_FRONT Returns")
    axis.set_xlabel("Forward distance from radar (m)")
    axis.set_ylabel("Left distance from radar (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.colorbar(points, ax=axis, label="Compensated forward velocity (m/s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Saved radar visualization: {output_path}")
    print(f"Radar point count: {radar.points.shape[1]}")


if __name__ == "__main__":
    main() #Make sure the output folder exists → open the dataset → take the first snapshot → load its front radar dots → pull out forward position, left position, and speed → plot them as colored dots on a top-down map with the car at the middle → save as a PNG → print the location and dot count