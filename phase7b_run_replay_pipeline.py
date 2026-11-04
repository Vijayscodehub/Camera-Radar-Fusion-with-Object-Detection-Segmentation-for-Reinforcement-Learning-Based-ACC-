import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PIPELINE_SCRIPTS = [
    "phase1_object_detection.py",
    "phase2_semantic_segmentation.py",
    "phase3_drivable_area.py",
    "phase4_radar_processing.py",
    "phase5a_radar_to_camera_projection.py",
    "phase5b_camera_radar_association.py",
    "phase6_scene_state.py",
    "phase6_rule_based_decision.py",
]
CSV_COLUMNS = [
    "frame_number",
    "sample_index",
    "camera_timestamp_us",
    "radar_camera_offset_ms",
    "detection_count",
    "radar_matched_detection_count",
    "lead_class",
    "lead_distance_m",
    "lead_relative_speed_mps",
    "ttc_s",
    "recommended_action",
]


def run_pipeline(sample_index: int) -> None:
    for script_name in PIPELINE_SCRIPTS:
        print(f"Running {script_name} for sample {sample_index}...")
        subprocess.run(
            [sys.executable, script_name, "--sample-index", str(sample_index)],
            check=True,
        )


def load_replay_row(frame: dict) -> dict:
    sample_index = frame["sample_index"]
    output_stem = f"cam_front_sample_{sample_index:03d}"
    fusion_path = Path("outputs/fusion") / f"{output_stem}_fused_obstacles.json"
    scene_state_path = Path("outputs/decision") / f"{output_stem}_scene_state.json"
    decision_path = Path("outputs/decision") / f"{output_stem}_rule_decision.json"

    fused_obstacles = json.loads(fusion_path.read_text())
    scene_state = json.loads(scene_state_path.read_text())
    decision = json.loads(decision_path.read_text())
    return {
        "frame_number": frame["frame_number"],
        "sample_index": sample_index,
        "camera_timestamp_us": frame["camera_timestamp_us"],
        "radar_camera_offset_ms": frame["radar_camera_offset_ms"],
        "detection_count": len(fused_obstacles),
        "radar_matched_detection_count": sum(
            obstacle["radar_matched"] for obstacle in fused_obstacles
        ),
        "lead_class": scene_state["lead_class"] or "none",
        "lead_distance_m": scene_state["lead_distance_m"],
        "lead_relative_speed_mps": scene_state["lead_relative_speed_mps"],
        "ttc_s": decision["ttc_s"],
        "recommended_action": decision["recommended_action"],
    }


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown_table(rows: list[dict], output_path: Path) -> None:
    columns = [
        ("Frame", "frame_number"),
        ("Sample", "sample_index"),
        ("Detections", "detection_count"),
        ("Radar matches", "radar_matched_detection_count"),
        ("Lead", "lead_class"),
        ("Distance (m)", "lead_distance_m"),
        ("Radar velocity (m/s)", "lead_relative_speed_mps"),
        ("TTC (s)", "ttc_s"),
        ("Action", "recommended_action"),
    ]
    lines = [
        "# Camera-Radar Replay Results",
        "",
        "| " + " | ".join(header for header, _ in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row[key]) for _, key in columns)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main(manifest_path: Path, max_frames: int | None, skip_pipeline: bool) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay manifest must contain a non-empty frames list.")
    selected_frames = frames[:max_frames] if max_frames is not None else frames
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"Max frames must be at least 1, got {max_frames}.")

    output_directory = Path("outputs/replay").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in selected_frames:
        sample_index = frame["sample_index"]
        if not skip_pipeline:
            run_pipeline(sample_index)
        rows.append(load_replay_row(frame))

    output_prefix = manifest_path.stem.removesuffix("_manifest")
    csv_path = output_directory / f"{output_prefix}_results.csv"
    markdown_path = output_directory / f"{output_prefix}_results.md"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_table(rows, markdown_path)

    print("\nReplay summary")
    for row in rows:
        print(
            f"Frame {row['frame_number']} / sample {row['sample_index']}: "
            f"lead={row['lead_class']}, distance={row['lead_distance_m']}, "
            f"action={row['recommended_action']}"
        )
    print(f"Saved replay CSV: {csv_path}")
    print(f"Saved replay Markdown: {markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Phase 1-6 pipeline across a continuous camera-radar replay."
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to a replay manifest created by phase7a_prepare_replay_sequence.py.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only this many leading manifest frames for a quick check.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Summarize existing Phase 1-6 outputs without rerunning them.",
    )
    arguments = parser.parse_args()
    main(arguments.manifest_path, arguments.max_frames, arguments.skip_pipeline)