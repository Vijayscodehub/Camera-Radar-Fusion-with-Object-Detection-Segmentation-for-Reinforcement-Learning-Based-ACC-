import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion #store a rotation


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
SAMPLE_INDEX = 95
POINT_COLOR = (0, 255, 255)

#This converts from "sensor's view" back to "a car's view.
def transform_points(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    rotation_matrix = Quaternion(rotation).rotation_matrix #rotation into a 3×3 number grid (matrix).
    translation_vector = np.array(translation).reshape(3, 1)
    return rotation_matrix @ points + translation_vector

#This converts from "car's view" back to "a sensor's view.
def inverse_transform_points(
    points: np.ndarray,
    rotation: list[float],
    translation: list[float],
) -> np.ndarray:
    rotation_matrix = Quaternion(rotation).rotation_matrix
    translation_vector = np.array(translation).reshape(3, 1)
    return rotation_matrix.T @ (points - translation_vector)


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    radar_directory = Path("outputs/radar").resolve()
    output_directory = Path("outputs/fusion").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    targets_path = radar_directory / f"radar_front_sample_{sample_index:03d}_targets.json"
    if not targets_path.is_file():
        raise FileNotFoundError(
            f"Missing filtered radar targets: {targets_path}. Run Phase 4 first."
        )
    targets = json.loads(targets_path.read_text())

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    radar_data = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    camera_path = dataset_root / camera_data["filename"]
    radar_path = dataset_root / radar_data["filename"]
    image = cv2.imread(str(camera_path))
    if image is None:
        raise FileNotFoundError(f"Could not load camera image: {camera_path}")

    radar = RadarPointCloud.from_file(str(radar_path))
    target_indices = np.array([target["radar_point_index"] for target in targets], dtype=int)
    points = radar.points[:3, target_indices]

    radar_calibration = nusc.get(
        "calibrated_sensor", radar_data["calibrated_sensor_token"]
    ) # where a sensor is bolted onto the car, and how it's tilted
    radar_pose = nusc.get("ego_pose", radar_data["ego_pose_token"])
    camera_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
    camera_calibration = nusc.get(
        "calibrated_sensor", camera_data["calibrated_sensor_token"]
    )

    points = transform_points(
        points, radar_calibration["rotation"], radar_calibration["translation"]
    ) #Radar → Car 
    points = transform_points(points, radar_pose["rotation"], radar_pose["translation"]) # Car → World
    points = inverse_transform_points(
        points, camera_pose["rotation"], camera_pose["translation"]
    ) #World → Car at the time of the camera photo
    points = inverse_transform_points(
        points,
        camera_calibration["rotation"],
        camera_calibration["translation"],
    ) #Car → Camera
    
    #Now the dots are in the camera's own 3D view: x = right, y = down, z = forward.
    image_points = view_points(
        points,
        np.array(camera_calibration["camera_intrinsic"]),
        normalize=True, #divides by depth, so far things look smaller and closer to center, just like real perspective
    ) #squash 3D into flat picture positions
    image_height, image_width = image.shape[:2]
    points_in_front = points[2] > 0
    points_inside_image = (
        (image_points[0] >= 0) #Pixel is not left of image
        & (image_points[0] < image_width) #Pixel is not right of image
        & (image_points[1] >= 0) #Pixel is not above image
        & (image_points[1] < image_height) #Pixel is not below image
    )
    keep_projected_points = points_in_front & points_inside_image

    projected_targets = []
    for local_index in np.flatnonzero(keep_projected_points): #gives the positions of the kept dots.
        target = targets[local_index].copy()
        #Round the pixel position to a whole number (you can't have half a pixel).
        pixel_x = round(float(image_points[0, local_index]))
        pixel_y = round(float(image_points[1, local_index]))
        target["pixel_x"] = pixel_x
        target["pixel_y"] = pixel_y
        projected_targets.append(target)
        cv2.circle(image, (pixel_x, pixel_y), 5, POINT_COLOR, -1)
        cv2.circle(image, (pixel_x, pixel_y), 7, (0, 0, 0), 1)

    output_stem = f"cam_front_sample_{sample_index:03d}"
    projected_targets_path = output_directory / f"{output_stem}_projected_radar_targets.json"
    projection_path = output_directory / f"{output_stem}_radar_projection.png"
    projected_targets_path.write_text(json.dumps(projected_targets, indent=2) + "\n")
    if not cv2.imwrite(str(projection_path), image):
        raise OSError(f"Could not save radar projection: {projection_path}")

    print(f"Sample index: {sample_index}")
    print(f"Filtered radar targets: {len(targets)}")
    print(f"Projected radar targets: {len(projected_targets)}")
    print(f"Saved projected targets JSON: {projected_targets_path}")
    print(f"Saved radar projection: {projection_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project filtered RADAR_FRONT targets onto the matching CAM_FRONT image."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    ) #Open a radar file → filter the dots → project the kept dots onto the matching camera photo → save a JSON file of the projected targets → save a picture of the projected targets
    arguments = parser.parse_args()
    main(arguments.sample_index)