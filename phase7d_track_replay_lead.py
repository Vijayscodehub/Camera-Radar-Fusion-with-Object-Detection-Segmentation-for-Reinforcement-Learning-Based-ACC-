import argparse
import csv
import json
from pathlib import Path


VEHICLE_CLASSES = {"car", "truck", "bus"}
MIN_IOU = 0.10
CSV_COLUMNS = [
    "frame_number",
    "sample_index",
    "lead_class",
    "lead_distance_m",
    "lead_relative_speed_mps",
    "bbox_iou_with_previous",
    "tracking_status",
]


def intersection_over_union(first_box: list[int], second_box: list[int]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


def vehicle_candidates(sample_index: int) -> list[dict]:
    output_stem = f"cam_front_sample_{sample_index:03d}"
    fusion_path = Path("outputs/fusion") / f"{output_stem}_fused_obstacles.json"
    if not fusion_path.is_file():
        raise FileNotFoundError(f"Missing fused obstacles: {fusion_path}")
    obstacles = json.loads(fusion_path.read_text())
    return [
        obstacle
        for obstacle in obstacles
        if obstacle["radar_matched"] and obstacle["class"] in VEHICLE_CLASSES
    ]


def first_lead(candidates: list[dict]) -> dict | None:
    return min(candidates, key=lambda candidate: candidate["distance_m"], default=None)


def tracked_row(
    frame: dict,
    lead: dict | None,
    iou: float | None,
    tracking_status: str,
) -> dict:
    return {
        "frame_number": frame["frame_number"],
        "sample_index": frame["sample_index"],
        "lead_class": lead["class"] if lead else "none",
        "lead_distance_m": lead["distance_m"] if lead else None,
        "lead_relative_speed_mps": lead["relative_speed_mps"] if lead else None,
        "bbox_iou_with_previous": round(iou, 3) if iou is not None else None,
        "tracking_status": tracking_status,
    }


def main(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay manifest must contain a non-empty frames list.")

    rows = []
    previous_lead = None
    for frame in frames:
        candidates = vehicle_candidates(frame["sample_index"])
        if previous_lead is None:
            lead = first_lead(candidates)
            status = "initialized" if lead else "no_vehicle_candidate"
            iou = None
        else:
            scored_candidates = [
                (intersection_over_union(previous_lead["bbox"], candidate["bbox"]), candidate)
                for candidate in candidates
            ]
            iou, lead = max(scored_candidates, default=(0.0, None), key=lambda item: item[0])
            if lead is None or iou < MIN_IOU:
                lead = None
                status = "tracking_lost"
            else:
                status = "tracked"

        rows.append(tracked_row(frame, lead, iou, status))
        previous_lead = lead

    output_directory = Path("outputs/replay").resolve()
    output_prefix = manifest_path.stem.removesuffix("_manifest")
    output_path = output_directory / f"{output_prefix}_tracked_lead.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    tracked_count = sum(row["tracking_status"] == "tracked" for row in rows)
    print(f"Replay frames: {len(rows)}")
    print(f"Consecutive tracked frames: {tracked_count}")
    print(f"Saved tracked lead CSV: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track one radar-matched lead vehicle across a replay manifest."
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to a replay manifest created by phase7a_prepare_replay_sequence.py.",
    )
    arguments = parser.parse_args()
    main(arguments.manifest_path) #It picks the closest radar-confirmed vehicle in the first frame, then in each later frame finds the vehicle whose box overlaps most with last frame's lead, and it calls it "lost" if the overlap is under 10%.