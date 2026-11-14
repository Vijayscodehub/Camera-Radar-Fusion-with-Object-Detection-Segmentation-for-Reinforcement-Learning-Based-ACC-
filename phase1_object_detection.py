import argparse
import json
from pathlib import Path

import cv2
from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.35
SAMPLE_INDEX = 95
RELEVANT_CLASSES = {"car", "truck", "bus", "person", "bicycle", "motorcycle"}


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    output_image_path = Path(
        f"outputs/detections/cam_front_sample_{sample_index:03d}_yolo11n.png"
    ).resolve()
    output_json_path = Path(
        f"outputs/detections/cam_front_sample_{sample_index:03d}_yolo11n.json"
    ).resolve()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image_path = dataset_root / camera_data["filename"]

    model = YOLO(MODEL_NAME)
    result = model(
        str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False,
    )[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load camera image: {image_path}")

    detections = []
    for box in result.boxes:
        class_name = result.names[int(box.cls[0])]
        if class_name not in RELEVANT_CLASSES:
            continue

        x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
        confidence = round(float(box.conf[0]), 4)
        detections.append(
            {
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            }
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_image_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())#turns the drawn image into PNG file data, and write_bytes saves it.
    output_json_path.write_text(json.dumps(detections, indent=2) + "\n")

    print(f"Input image: {camera_data['filename']}")
    print(f"Sample index: {sample_index}")
    print("Inference device: CPU")
    print(f"Relevant detections: {len(detections)}")
    print(f"Saved annotated image: {output_image_path}")
    print(f"Saved detections JSON: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CPU-only YOLO11n detection on a nuScenes CAM_FRONT image."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    )
    arguments = parser.parse_args()
    main(arguments.sample_index)