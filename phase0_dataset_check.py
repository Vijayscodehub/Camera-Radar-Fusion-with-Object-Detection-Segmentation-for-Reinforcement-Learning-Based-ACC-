from pathlib import Path

import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"


def main() -> None:
    dataset_root = DATA_ROOT.resolve()
    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    sample = nusc.sample[0]

    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    radar_data = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    camera_path = dataset_root / camera_data["filename"]
    radar_path = dataset_root / radar_data["filename"]
    image = cv2.imread(str(camera_path))
    radar_points = RadarPointCloud.from_file(str(radar_path))
    camera_calibration = nusc.get(
        "calibrated_sensor", camera_data["calibrated_sensor_token"]
    )
    radar_calibration = nusc.get(
        "calibrated_sensor", radar_data["calibrated_sensor_token"]
    )
    camera_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])

    if image is None:
        raise FileNotFoundError(f"Could not load camera image: {camera_path}")

    print(f"Scenes loaded: {len(nusc.scene)}")
    print(f"Sample token: {sample['token']}")
    print(f"Sample timestamp: {sample['timestamp']}")
    print(f"Camera file: {camera_data['filename']}")
    print(f"Camera image shape (height, width, channels): {image.shape}")
    print(f"Radar file: {radar_data['filename']}")
    print(f"Radar point count: {radar_points.points.shape[1]}")
    print(f"Camera timestamp: {camera_data['timestamp']}")
    print(f"Radar timestamp: {radar_data['timestamp']}")
    print(f"Camera intrinsic matrix: {camera_calibration['camera_intrinsic']}")
    print(f"Camera translation in ego frame: {camera_calibration['translation']}")
    print(f"Radar translation in ego frame: {radar_calibration['translation']}")
    print(f"Camera ego pose translation: {camera_pose['translation']}")
    print(f"Annotation count: {len(sample['anns'])}")


if __name__ == "__main__":
    main()