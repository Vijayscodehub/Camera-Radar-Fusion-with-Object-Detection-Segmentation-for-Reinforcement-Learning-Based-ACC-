from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
OUTPUT_PATH = Path("outputs/phase0/cam_front_ground_truth.png")


def main() -> None:
    dataset_root = DATA_ROOT.resolve()
    output_path = OUTPUT_PATH.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    sample = nusc.sample[0]
    camera_token = sample["data"]["CAM_FRONT"]

    nusc.render_sample_data(
        camera_token,
        with_anns=True,
        out_path=str(output_path),
    )

    print(f"Saved ground-truth annotation image: {output_path}")


if __name__ == "__main__":
    main()