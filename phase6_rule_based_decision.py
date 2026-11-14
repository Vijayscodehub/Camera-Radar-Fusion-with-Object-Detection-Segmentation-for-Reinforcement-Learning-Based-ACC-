import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


DATA_ROOT = Path("data/nuscenes_dataset/v1.0-mini")
DATASET_VERSION = "v1.0-mini"
SAMPLE_INDEX = 95
BRAKE_DISTANCE_M = 12.0
MAINTAIN_DISTANCE_M = 30.0
BRAKE_TTC_S = 4.0
DRIVABLE_OVERLAY_ALPHA = 0.35
DRIVABLE_COLOR = (0, 180, 0)
ACTION_COLORS = {
    "BRAKE": (0, 0, 255),
    "MAINTAIN": (0, 255, 255),
    "ACCELERATE": (0, 255, 0),
}


def calculate_ttc(distance_m: float | None, relative_speed_mps: float | None) -> float | None:
    if distance_m is None or relative_speed_mps is None or relative_speed_mps <= 0:
        return None
    return round(distance_m / relative_speed_mps, 2)


def decide_action(scene_state: dict) -> tuple[str, str, float | None]:
    distance_m = scene_state["lead_distance_m"]
    relative_speed_mps = scene_state["lead_relative_speed_mps"]
    ttc_s = calculate_ttc(distance_m, relative_speed_mps)

    if scene_state["lead_present"] and distance_m <= BRAKE_DISTANCE_M:
        return "BRAKE", f"Lead object is within {BRAKE_DISTANCE_M:.0f} m.", ttc_s
    if scene_state["lead_present"] and ttc_s is not None and ttc_s <= BRAKE_TTC_S:
        return "BRAKE", f"Lead-object TTC is {ttc_s:.2f} s.", ttc_s
    if not scene_state["drivable_ahead"]:
        return "MAINTAIN", "Drivable road is not confirmed in the lookahead region.", ttc_s
    if scene_state["lead_present"] and distance_m <= MAINTAIN_DISTANCE_M:
        return "MAINTAIN", f"Lead object is within {MAINTAIN_DISTANCE_M:.0f} m.", ttc_s
    if not scene_state["lead_present"]:
        return "ACCELERATE", "No radar-matched lead vehicle is on the drivable area.", ttc_s
    return "ACCELERATE", "Lead object is beyond the maintain-distance threshold.", ttc_s


def create_decision_overlay(
    image: np.ndarray, drivable_mask: np.ndarray, scene_state: dict, action: str
) -> np.ndarray:
    overlay = image.copy()
    overlay[drivable_mask] = (
        (1 - DRIVABLE_OVERLAY_ALPHA) * image[drivable_mask]
        + DRIVABLE_OVERLAY_ALPHA * np.array(DRIVABLE_COLOR)
    ).astype(np.uint8)

    action_color = ACTION_COLORS[action]
    if scene_state["lead_present"]:
        x1, y1, x2, y2 = scene_state["lead_bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), action_color, 3)
        label = (
            f"Lead {scene_state['lead_class']}: "
            f"{scene_state['lead_distance_m']:.1f}m"
        )
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 55)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            action_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"Action: {action}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        action_color,
        3,
        cv2.LINE_AA,
    )
    return overlay


def main(sample_index: int) -> None:
    dataset_root = DATA_ROOT.resolve()
    decision_directory = Path("outputs/decision").resolve()
    drivable_directory = Path("outputs/drivable_area").resolve()
    decision_directory.mkdir(parents=True, exist_ok=True)

    output_stem = f"cam_front_sample_{sample_index:03d}"
    scene_state_path = decision_directory / f"{output_stem}_scene_state.json"
    drivable_mask_path = drivable_directory / f"{output_stem}_drivable_binary.png"
    for required_path, description in [
        (scene_state_path, "scene state"),
        (drivable_mask_path, "drivable-area binary mask"),
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing {description}: {required_path}.")

    scene_state = json.loads(scene_state_path.read_text())
    action, decision_reason, ttc_s = decide_action(scene_state)
    decision = {
        "sample_index": sample_index,
        "recommended_action": action,
        "decision_reason": decision_reason,
        "lead_class": scene_state["lead_class"],
        "lead_distance_m": scene_state["lead_distance_m"],
        "lead_relative_speed_mps": scene_state["lead_relative_speed_mps"],
        "ttc_s": ttc_s,
        "drivable_ahead": scene_state["drivable_ahead"],
        "ego_speed_mps": scene_state["ego_speed_mps"],
    }

    nusc = NuScenes(version=DATASET_VERSION, dataroot=str(dataset_root), verbose=False)
    if not 0 <= sample_index < len(nusc.sample):
        raise ValueError(
            f"Sample index must be between 0 and {len(nusc.sample) - 1}, got {sample_index}."
        )
    sample = nusc.sample[sample_index]
    camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    image_path = dataset_root / camera_data["filename"]
    image = cv2.imread(str(image_path))
    drivable_mask = cv2.imread(str(drivable_mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load camera image: {image_path}")
    if drivable_mask is None:
        raise FileNotFoundError(f"Could not load drivable mask: {drivable_mask_path}")
    if image.shape[:2] != drivable_mask.shape:
        raise ValueError("The camera image and drivable mask have different dimensions.")

    overlay = create_decision_overlay(image, drivable_mask > 0, scene_state, action)
    decision_path = decision_directory / f"{output_stem}_rule_decision.json"
    overlay_path = decision_directory / f"{output_stem}_decision_overlay.png"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not save decision overlay: {overlay_path}")

    print(f"Sample index: {sample_index}")
    print(f"Recommended action: {action}")
    print(f"Decision reason: {decision_reason}")
    print(f"TTC: {ttc_s if ttc_s is not None else 'not applicable'}")
    print(f"Saved decision JSON: {decision_path}")
    print(f"Saved decision overlay: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply an explainable rule-based decision to a Phase 6 scene state."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    ) #Open the scene state JSON → open the drivable area binary mask → apply the rule-based decision logic → save a decision JSON and an overlay image showing the recommended action
    arguments = parser.parse_args()
    main(arguments.sample_index)
    
    #distance_m is hypo of the x and y coordinates of the radar point. because we want actual sideways dists to the target. and also we use the median of the points in the bb for the lead vehicle