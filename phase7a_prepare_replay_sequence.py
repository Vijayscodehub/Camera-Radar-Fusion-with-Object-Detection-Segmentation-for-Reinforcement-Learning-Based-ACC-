import argparse
import json
from pathlib import Path

from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
START_SAMPLE_INDEX = 9
FRAME_COUNT = 10


def replay_frame(
    nusc: NuScenes,
    sample: dict,
    sample_indices_by_token: dict[str, int],
    frame_number: int,
) -> dict:
    if "CAM_FRONT" not in sample["data"] or "RADAR_FRONT" not in sample["data"]:
        raise KeyError("Each replay frame must contain CAM_FRONT and RADAR_FRONT.")

    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    radar_data = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    return {
        "frame_number": frame_number,
        "sample_index": sample_indices_by_token[sample["token"]],
        "sample_token": sample["token"],
        "sample_timestamp_us": sample["timestamp"],
        "camera_filename": camera_data["filename"],
        "camera_timestamp_us": camera_data["timestamp"],
        "radar_filename": radar_data["filename"],
        "radar_timestamp_us": radar_data["timestamp"],
        "radar_camera_offset_ms": round(
            (radar_data["timestamp"] - camera_data["timestamp"]) / 1_000, 3
        ),
    }


def timestamp_range_samples(
    nusc: NuScenes,
    start_camera_timestamp: int,
    end_camera_timestamp: int,
) -> list[dict]:
    if start_camera_timestamp > end_camera_timestamp:
        raise ValueError("Start camera timestamp must not be after end camera timestamp.")

    camera_keyframes = sorted(
        (
            sample_data
            for sample_data in nusc.sample_data
            if sample_data["channel"] == "CAM_FRONT"
            and sample_data["is_key_frame"]
            and start_camera_timestamp <= sample_data["timestamp"] <= end_camera_timestamp
        ),
        key=lambda sample_data: sample_data["timestamp"],
    )
    if not camera_keyframes:
        raise ValueError("No CAM_FRONT keyframes were found in the requested timestamp range.")

    samples = [nusc.get("sample", keyframe["sample_token"]) for keyframe in camera_keyframes]
    scene_tokens = {sample["scene_token"] for sample in samples}
    if len(scene_tokens) != 1:
        raise ValueError("Timestamp range must contain keyframes from exactly one scene.")
    for previous_sample, current_sample in zip(samples, samples[1:]): #pairs up neighbours
        if previous_sample["next"] != current_sample["token"]:
            raise ValueError("Selected keyframes are not consecutive within their scene.")
    return samples


def main(
    start_sample_index: int,
    frame_count: int,
    start_camera_timestamp: int | None,
    end_camera_timestamp: int | None,
) -> None:
    timestamp_mode = start_camera_timestamp is not None or end_camera_timestamp is not None
    if timestamp_mode and (start_camera_timestamp is None or end_camera_timestamp is None):
        raise ValueError(
            "Both start and end camera timestamps are required for timestamp-range mode."
        )
    if not timestamp_mode and frame_count < 1:
        raise ValueError(f"Frame count must be at least 1, got {frame_count}.")

    dataset_root = DATA_ROOT.resolve()
    output_directory = Path("outputs/replay").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)

    if not 0 <= start_sample_index < len(nusc.sample):
        raise ValueError(
            "Start sample index must be between "
            f"0 and {len(nusc.sample) - 1}, got {start_sample_index}."
        )

    sample_indices_by_token = {
        sample["token"]: index for index, sample in enumerate(nusc.sample)
    }
    if timestamp_mode:
        samples = timestamp_range_samples(
            nusc, start_camera_timestamp, end_camera_timestamp
        )
    else:
        start_sample = nusc.sample[start_sample_index]
        samples = []
        current_sample = start_sample
        for _ in range(frame_count):
            samples.append(current_sample)
            if not current_sample["next"]:
                break
            current_sample = nusc.get("sample", current_sample["next"])

    start_sample = samples[0]
    scene = nusc.get("scene", start_sample["scene_token"])
    sequence = [
        replay_frame(nusc, sample, sample_indices_by_token, frame_number)
        for frame_number, sample in enumerate(samples)
    ]

    manifest = {
        "scene_name": scene["name"],
        "selection_mode": "camera_timestamp_range" if timestamp_mode else "sample_index",
        "start_sample_index": sequence[0]["sample_index"],
        "actual_frame_count": len(sequence),
        "frames": sequence,
    }
    if timestamp_mode:
        manifest["camera_timestamp_range_us"] = [
            start_camera_timestamp,
            end_camera_timestamp,
        ]
        output_name = (
            f"{scene['name']}_camera_{start_camera_timestamp}_to_"
            f"{end_camera_timestamp}_manifest.json"
        )
    else:
        manifest["requested_frame_count"] = frame_count
        output_name = f"{scene['name']}_from_sample_{start_sample_index:03d}_manifest.json"

    output_path = output_directory / output_name
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Scene: {scene['name']}")
    print(f"Start sample index: {sequence[0]['sample_index']}")
    print(f"Replay frames: {len(sequence)}")
    print(f"Saved replay manifest: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a continuous camera-radar replay manifest from a nuScenes scene."
    )
    parser.add_argument(
        "--start-sample-index",
        type=int,
        default=START_SAMPLE_INDEX,
        help=(
            "Global nuScenes sample index for the first replay frame "
            f"(default: {START_SAMPLE_INDEX})."
        ),
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=FRAME_COUNT,
        help=f"Number of consecutive scene frames to include (default: {FRAME_COUNT}).",
    )
    parser.add_argument(
        "--start-camera-timestamp",
        type=int,
        help="First CAM_FRONT timestamp in the inclusive keyframe range.",
    )
    parser.add_argument(
        "--end-camera-timestamp",
        type=int,
        help="Last CAM_FRONT timestamp in the inclusive keyframe range.",
    )
    arguments = parser.parse_args()
    main(
        arguments.start_sample_index,
        arguments.frame_count,
        arguments.start_camera_timestamp,
        arguments.end_camera_timestamp,
    ) #Create a continuous camera-radar replay manifest from a nuScenes scene.