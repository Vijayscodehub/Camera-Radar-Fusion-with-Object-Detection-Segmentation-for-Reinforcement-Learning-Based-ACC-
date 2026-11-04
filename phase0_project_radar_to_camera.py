from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
OUTPUT_PATH = Path("outputs/phase0/cam_front_radar_projection.png")


def main() -> None:
    dataset_root = DATA_ROOT.resolve()
    output_path = OUTPUT_PATH.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    sample = nusc.sample[0]
    camera_token = sample["data"]["CAM_FRONT"]
    radar_token = sample["data"]["RADAR_FRONT"]

    image_points, point_ranges_m, image = nusc.explorer.map_pointcloud_to_image(
        radar_token,
        camera_token,
    )

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.imshow(image)
    scatter = axis.scatter(
        image_points[0],
        image_points[1],
        c=point_ranges_m,
        cmap="turbo",
        s=45,
        edgecolors="black",
        linewidths=0.4,
    )
    axis.set_title("RADAR_FRONT Points Projected onto CAM_FRONT")
    axis.set_axis_off()
    figure.colorbar(scatter, ax=axis, label="Radar range (m)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved radar-camera projection: {output_path}")
    print(f"Projected radar point count: {image_points.shape[1]}")


if __name__ == "__main__":
    main()