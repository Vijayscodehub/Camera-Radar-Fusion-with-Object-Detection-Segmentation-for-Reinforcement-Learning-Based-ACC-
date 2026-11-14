import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SAMPLE_INDEX = 95
VEHICLE_CLASSES = {"car", "truck", "bus"}
EGO_SPEED_MPS = 13.9 #  nuScenes dataset car 50 km/h - 13.9 m/s.
##Uses the detected car's bounding box to check if the bottom of the car is on the road, and if there is road ahead of the car.
FOOTPOINT_WIDTH_FRACTION = 0.30 # 30% of the width of the bounding box at the bottom of car where the car touches the road
FOOTPOINT_HEIGHT_PIXELS = 12 #strip 12 pixels tall in that 30% region

##Looks at the road ahead of our car, Uses a fixed region of the camera image not the bounding box of the detected car.
MIN_FOOTPOINT_DRIVABLE_FRACTION = 0.20 #20% of that strip must be road
LOOKAHEAD_X_MIN_FRACTION = 0.40 #horizontal area we want to look at middle part of the picture (40% till 60%)
LOOKAHEAD_X_MAX_FRACTION = 0.60
LOOKAHEAD_Y_MIN_FRACTION = 0.55 #lower part of the picture (55% till 90% down)
LOOKAHEAD_Y_MAX_FRACTION = 0.90
MIN_LOOKAHEAD_DRIVABLE_FRACTION = 0.05 # 5% of the above fraction area must be road

def footpoint_drivable_fraction(
    drivable_mask: np.ndarray, bounding_box: list[int]
) -> float:
    image_height, image_width = drivable_mask.shape
    x1, y1, x2, y2 = bounding_box
    box_width = max(x2 - x1, 1) #sanity check
    half_footpoint_width = max(
        round(box_width * FOOTPOINT_WIDTH_FRACTION / 2), 1
    )
    center_x = round((x1 + x2) / 2) #centre of the detected car

    footpoint_x1 = max(center_x - half_footpoint_width, 0)
    footpoint_x2 = min(center_x + half_footpoint_width + 1, image_width)
    footpoint_y1 = max(y2 - FOOTPOINT_HEIGHT_PIXELS + 1, y1, 0)
    footpoint_y2 = min(y2 + 1, image_height)
    footpoint_region = drivable_mask[
        footpoint_y1:footpoint_y2, footpoint_x1:footpoint_x2
    ] #array slicing  #From the road image, give me only the pixels inside this small rectangles

    if footpoint_region.size == 0:
        return 0.0
    return float(np.count_nonzero(footpoint_region) / footpoint_region.size) # % of the pixels in that small rectangle that are road


def drivable_ahead_fraction(drivable_mask: np.ndarray) -> float:
    image_height, image_width = drivable_mask.shape
    region = drivable_mask[
        round(image_height * LOOKAHEAD_Y_MIN_FRACTION) : round(
            image_height * LOOKAHEAD_Y_MAX_FRACTION
        ),
        round(image_width * LOOKAHEAD_X_MIN_FRACTION) : round(
            image_width * LOOKAHEAD_X_MAX_FRACTION
        ),
    ]

    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region) / region.size)


def main(sample_index: int) -> None:
    fusion_directory = Path("outputs/fusion").resolve()
    drivable_directory = Path("outputs/drivable_area").resolve()
    output_directory = Path("outputs/decision").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    output_stem = f"cam_front_sample_{sample_index:03d}"
    fused_obstacles_path = fusion_directory / f"{output_stem}_fused_obstacles.json"
    drivable_mask_path = drivable_directory / f"{output_stem}_drivable_binary.png"
    for required_path, description in [
        (fused_obstacles_path, "fused obstacles"),
        (drivable_mask_path, "drivable-area binary mask"),
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing {description}: {required_path}.")

    fused_obstacles = json.loads(fused_obstacles_path.read_text())
    drivable_mask = cv2.imread(str(drivable_mask_path), cv2.IMREAD_GRAYSCALE)
    if drivable_mask is None:
        raise FileNotFoundError(f"Could not load drivable mask: {drivable_mask_path}")
    drivable_mask = drivable_mask > 0

    candidates = []
    for obstacle in fused_obstacles:
        if not obstacle["radar_matched"] or obstacle["class"] not in VEHICLE_CLASSES:
            continue

        overlap_fraction = footpoint_drivable_fraction(
            drivable_mask, obstacle["bbox"]
        )
        if overlap_fraction < MIN_FOOTPOINT_DRIVABLE_FRACTION:
            continue

        candidate = obstacle.copy()
        candidate["footpoint_drivable_fraction"] = round(overlap_fraction, 3)
        candidates.append(candidate)

    lead_object = min(candidates, key=lambda candidate: candidate["distance_m"], default=None) # Select the closest vehicle candidate
    lookahead_fraction = drivable_ahead_fraction(drivable_mask)
    scene_state = {
        "sample_index": sample_index,
        "lead_present": lead_object is not None,
        "lead_class": lead_object["class"] if lead_object else None,
        "lead_bbox": lead_object["bbox"] if lead_object else None,
        "lead_distance_m": lead_object["distance_m"] if lead_object else None,
        "lead_relative_speed_mps": (
            lead_object["relative_speed_mps"] if lead_object else None
        ),
        "lead_confidence": lead_object["confidence"] if lead_object else None,
        "lead_radar_point_count": (
            lead_object["radar_point_count"] if lead_object else 0
        ),
        "lead_footpoint_drivable_fraction": (
            lead_object["footpoint_drivable_fraction"] if lead_object else None
        ),
        "lead_candidate_count": len(candidates),
        "drivable_ahead": lookahead_fraction >= MIN_LOOKAHEAD_DRIVABLE_FRACTION,
        "drivable_ahead_fraction": round(lookahead_fraction, 3),
        "ego_speed_mps": EGO_SPEED_MPS,
    }

    output_path = output_directory / f"{output_stem}_scene_state.json"
    output_path.write_text(json.dumps(scene_state, indent=2) + "\n")

    print(f"Sample index: {sample_index}")
    print(f"Vehicle candidates on drivable area: {len(candidates)}")
    print(f"Drivable ahead: {scene_state['drivable_ahead']}")
    if lead_object:
        print(
            f"Selected lead: {scene_state['lead_class']} at "
            f"{scene_state['lead_distance_m']:.2f} m"
        )
    else:
        print("Selected lead: none")
    print(f"Saved scene state: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a lead-object scene state from fusion and drivable-area outputs."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help=f"nuScenes sample index to process (default: {SAMPLE_INDEX}).",
    ) ##Open the fused obstacles JSON → open the drivable area binary mask → check which of the fused obstacles are on the road (at least 20% road under the vehicle's feet) → check if there is road ahead of our car (min percent of road - 5%) → save a scene state JSON file

    arguments = parser.parse_args()
    main(arguments.sample_index)