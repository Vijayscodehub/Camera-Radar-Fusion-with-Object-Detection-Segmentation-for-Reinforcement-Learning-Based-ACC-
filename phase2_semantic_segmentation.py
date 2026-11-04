import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
MODEL_DIRECTORY = Path("models/segformer-b0-cityscapes")
SAMPLE_INDEX = 95
OVERLAY_ALPHA = 0.45

CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    model_directory = MODEL_DIRECTORY.resolve()
    output_directory = Path("outputs/segmentation").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )

    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image_path = dataset_root / camera_data["filename"]
    image = Image.open(image_path).convert("RGB")

    required_model_files = [
        model_directory / "config.json",
        model_directory / "preprocessor_config.json",
        model_directory / "pytorch_model.bin",
    ]
    missing_model_files = [path.name for path in required_model_files if not path.is_file()]
    if missing_model_files:
        missing_files = ", ".join(missing_model_files)
        raise FileNotFoundError(
            f"Missing model files in {model_directory}: {missing_files}."
        )

    processor = AutoImageProcessor.from_pretrained(model_directory, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_directory,
        local_files_only=True,
    ).to("cpu")
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        model_outputs = model(**inputs)

    class_mask = processor.post_process_semantic_segmentation(
        model_outputs,
        target_sizes=[(image.height, image.width)],
    )[0].numpy().astype(np.uint8)
    colored_mask = CITYSCAPES_COLORS[class_mask]
    original_image = np.array(image)
    overlay = cv2.addWeighted(original_image, 1 - OVERLAY_ALPHA, colored_mask, OVERLAY_ALPHA, 0)

    output_stem = f"cam_front_sample_{sample_index:03d}_segformer_b0"
    class_mask_path = output_directory / f"{output_stem}_class_ids.png"
    colored_mask_path = output_directory / f"{output_stem}_colored.png"
    overlay_path = output_directory / f"{output_stem}_overlay.png"

    Image.fromarray(class_mask).save(class_mask_path)
    Image.fromarray(colored_mask).save(colored_mask_path)
    Image.fromarray(overlay).save(overlay_path)

    print(f"Input image: {camera_data['filename']}")
    print(f"Sample index: {sample_index}")
    print("Inference device: CPU")
    print(f"Class IDs found: {sorted(np.unique(class_mask).tolist())}")
    print(f"Saved class-ID mask: {class_mask_path}")
    print(f"Saved colored mask: {colored_mask_path}")
    print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CPU-only SegFormer-B0 segmentation on a nuScenes CAM_FRONT image."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    )
    arguments = parser.parse_args()
    main(arguments.sample_index)