import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
SAMPLE_INDEX = 95
BOX_COLOR = (0, 255, 0)
MATCHED_POINT_COLOR = (0, 255, 255)


def point_is_inside_box(point: dict, bounding_box: list[int]) -> bool:
    x1, y1, x2, y2 = bounding_box
    return x1 <= point["pixel_x"] <= x2 and y1 <= point["pixel_y"] <= y2


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    detections_directory = Path("outputs/detections").resolve()
    fusion_directory = Path("outputs/fusion").resolve()
    fusion_directory.mkdir(parents=True, exist_ok=True)

    detections_path = detections_directory / f"cam_front_sample_{sample_index:03d}_yolo11n.json"
    projected_targets_path = (
        fusion_directory / f"cam_front_sample_{sample_index:03d}_projected_radar_targets.json"
    )
    for required_path, description in [
        (detections_path, "YOLO detections"),
        (projected_targets_path, "projected radar targets"),
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing {description}: {required_path}.")

    detections = json.loads(detections_path.read_text())
    projected_targets = json.loads(projected_targets_path.read_text())

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image_path = dataset_root / camera_data["filename"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load camera image: {image_path}")

    fused_obstacles = []
    for detection in detections:
        matched_targets = [
            target
            for target in projected_targets
            if point_is_inside_box(target, detection["bbox"])
        ]
        fused_obstacle = detection.copy()
        fused_obstacle["radar_matched"] = bool(matched_targets)
        fused_obstacle["radar_point_count"] = len(matched_targets)

        if matched_targets:
            fused_obstacle["distance_m"] = round(
                float(np.median([target["distance_m"] for target in matched_targets])), 2
            )
            fused_obstacle["relative_speed_mps"] = round(
                float(
                    np.median(
                        [target["relative_velocity_mps"] for target in matched_targets]
                    )
                ),
                2,
            )
            fused_obstacle["radar_point_indices"] = [
                target["radar_point_index"] for target in matched_targets
            ]
        else:
            fused_obstacle["distance_m"] = None
            fused_obstacle["relative_speed_mps"] = None
            fused_obstacle["radar_point_indices"] = []

        fused_obstacles.append(fused_obstacle)

        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        if matched_targets:
            label = (
                f"{detection['class']} {fused_obstacle['distance_m']:.1f}m "
                f"{fused_obstacle['relative_speed_mps']:.1f}m/s"
            )
            for target in matched_targets:
                cv2.circle(
                    image,
                    (target["pixel_x"], target["pixel_y"]),
                    6,
                    MATCHED_POINT_COLOR,
                    -1,
                )
        else:
            label = f"{detection['class']} no radar match"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    output_stem = f"cam_front_sample_{sample_index:03d}"
    fused_obstacles_path = fusion_directory / f"{output_stem}_fused_obstacles.json"
    overlay_path = fusion_directory / f"{output_stem}_fusion_overlay.png"
    fused_obstacles_path.write_text(json.dumps(fused_obstacles, indent=2) + "\n")
    if not cv2.imwrite(str(overlay_path), image):
        raise OSError(f"Could not save fusion overlay: {overlay_path}")

    matched_count = sum(obstacle["radar_matched"] for obstacle in fused_obstacles)
    print(f"Sample index: {sample_index}")
    print(f"YOLO detections: {len(detections)}")
    print(f"Radar-matched detections: {matched_count}")
    print(f"Saved fused obstacles JSON: {fused_obstacles_path}")
    print(f"Saved fusion overlay: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Associate projected radar targets with YOLO detections."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    )
    arguments = parser.parse_args()
    main(arguments.sample_index)