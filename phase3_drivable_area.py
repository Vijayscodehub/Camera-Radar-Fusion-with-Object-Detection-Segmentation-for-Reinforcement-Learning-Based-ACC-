import argparse
import json
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from PIL import Image


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
SAMPLE_INDEX = 95
ROAD_CLASS_ID = 0 #means "road" (0).
OVERLAY_ALPHA = 0.45
DRIVABLE_COLOR = np.array([0, 255, 0], dtype=np.uint8)


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    segmentation_directory = Path("outputs/segmentation").resolve()
    output_directory = Path("outputs/drivable_area").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    class_mask_path = (
        segmentation_directory
        / f"cam_front_sample_{sample_index:03d}_segformer_b0_class_ids.png"
    )
    if not class_mask_path.is_file():
        raise FileNotFoundError(
            f"Missing segmentation class-ID mask: {class_mask_path}. "
            "Run Phase 2 for this sample first."
        )

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image_path = dataset_root / camera_data["filename"]
    original_image = np.array(Image.open(image_path).convert("RGB"))
    class_mask = np.array(Image.open(class_mask_path))

    if class_mask.shape != original_image.shape[:2]:
        raise ValueError(
            "The segmentation mask and camera image have different dimensions."
        )

    drivable_mask = class_mask == ROAD_CLASS_ID #Asks every pixel, "Are you 0 (road)?"
    binary_mask = np.where(drivable_mask, 255, 0).astype(np.uint8)
    overlay = original_image.copy()
    overlay[drivable_mask] = (
        (1 - OVERLAY_ALPHA) * original_image[drivable_mask]
        + OVERLAY_ALPHA * DRIVABLE_COLOR
    ).astype(np.uint8)

    output_stem = f"cam_front_sample_{sample_index:03d}_drivable"
    binary_mask_path = output_directory / f"{output_stem}_binary.png"
    overlay_path = output_directory / f"{output_stem}_overlay.png"
    stats_path = output_directory / f"{output_stem}_stats.json"

    drivable_pixels = int(np.count_nonzero(drivable_mask))
    total_pixels = int(drivable_mask.size)
    statistics = {
        "sample_index": sample_index,
        "road_class_id": ROAD_CLASS_ID,
        "drivable_pixels": drivable_pixels,
        "total_pixels": total_pixels,
        "drivable_percentage": round(100 * drivable_pixels / total_pixels, 2),
    }

    Image.fromarray(binary_mask).save(binary_mask_path)
    Image.fromarray(overlay).save(overlay_path)
    stats_path.write_text(json.dumps(statistics, indent=2) + "\n")

    print(f"Input class-ID mask: {class_mask_path}")
    print(f"Sample index: {sample_index}")
    print(f"Drivable pixels: {drivable_pixels}/{total_pixels}")
    print(f"Drivable percentage: {statistics['drivable_percentage']}%")
    print(f"Saved binary mask: {binary_mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved statistics: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a binary drivable-area mask from SegFormer road predictions."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    ) #Open a photo → open the SegFormer class-ID mask → ask every pixel, "Are you 0 (road)?" → color the road pixels green → save a binary mask and an overlay
    arguments = parser.parse_args()
    main(arguments.sample_index)