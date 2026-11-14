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
        with_anns=True, #also draw the annotations" (the labeled boxes).
        out_path=str(output_path),
    )

    print(f"Saved ground-truth annotation image: {output_path}")


if __name__ == "__main__":
    main() #Make sure the output folder exists → open the dataset → take the first snapshot → find its front-camera reading → ask nuScenes to draw that photo with the labeled boxes → save it as a PNG → print where it was saved.