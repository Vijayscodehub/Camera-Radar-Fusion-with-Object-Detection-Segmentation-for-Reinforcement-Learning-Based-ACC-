import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SAMPLE_INDICES = [0, 50, 95]
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
    "sample_index",
    "detection_count",
    "radar_matched_detection_count",
    "lead_class",
    "lead_distance_m",
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


def load_evaluation_row(sample_index: int) -> dict:
    output_stem = f"cam_front_sample_{sample_index:03d}"
    fusion_path = Path("outputs/fusion") / f"{output_stem}_fused_obstacles.json"
    scene_state_path = Path("outputs/decision") / f"{output_stem}_scene_state.json"
    decision_path = Path("outputs/decision") / f"{output_stem}_rule_decision.json"

    fused_obstacles = json.loads(fusion_path.read_text())
    scene_state = json.loads(scene_state_path.read_text())
    decision = json.loads(decision_path.read_text())

    return {
        "sample_index": sample_index,
        "detection_count": len(fused_obstacles),
        "radar_matched_detection_count": sum(
            obstacle["radar_matched"] for obstacle in fused_obstacles
        ),
        "lead_class": scene_state["lead_class"] or "none",
        "lead_distance_m": scene_state["lead_distance_m"],
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
    headers = [
        "Sample",
        "Detections",
        "Radar matches",
        "Lead object",
        "Lead distance (m)",
        "TTC (s)",
        "Action",
    ]
    lines = [
        "# Multi-Sample Phase 1-6 Evaluation",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row[column]) for column in CSV_COLUMNS)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main(sample_indices: list[int], skip_pipeline: bool) -> None:
    output_directory = Path("outputs/evaluation").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_index in sample_indices:
        if not skip_pipeline:
            run_pipeline(sample_index)
        rows.append(load_evaluation_row(sample_index))

    csv_path = output_directory / "phase1_to_6_evaluation.csv"
    markdown_path = output_directory / "phase1_to_6_evaluation.md"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_table(rows, markdown_path)

    print("\nEvaluation summary")
    for row in rows:
        print(
            f"Sample {row['sample_index']}: {row['detection_count']} detections, "
            f"{row['radar_matched_detection_count']} radar matches, "
            f"lead={row['lead_class']}, action={row['recommended_action']}"
        )
    print(f"Saved CSV table: {csv_path}")
    print(f"Saved Markdown table: {markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phases 1-6 over multiple nuScenes samples and summarize results."
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_INDICES,
        help=(
            "nuScenes sample indices to evaluate "
            f"(default: {' '.join(map(str, DEFAULT_SAMPLE_INDICES))})."
        ),
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Summarize existing Phase 1-6 outputs without rerunning the pipeline.",
    )
    arguments = parser.parse_args()
    main(arguments.sample_indices, arguments.skip_pipeline)