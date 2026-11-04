# Phase 6: Scene State and Rule-Based Decision

## Goal

Convert the Phase 5 fused obstacles and Phase 3 drivable-area result into one small, explainable scene state. Use that state to recommend one action:

```text
BRAKE
MAINTAIN
ACCELERATE
```

This is an offline recommendation for the selected nuScenes frame. It does not control a real vehicle or run in real time.

```text
Fused obstacles + drivable-area mask
-> choose the likely lead object
-> create numerical scene state
-> apply rule-based controller
-> action and explanation
```

## Inputs

```text
Phase 3
outputs/drivable_area/cam_front_sample_095_drivable_binary.png
outputs/drivable_area/cam_front_sample_095_drivable_stats.json

Phase 5
outputs/fusion/cam_front_sample_095_fused_obstacles.json
```

The fused-obstacles JSON supplies object class, image bounding box, radar range, compensated relative velocity, YOLO confidence, and radar-match information. The binary drivable mask supplies road context in the same image coordinates.

## Step 6A: Identify Drivable Objects

For each radar-matched vehicle detection (`car`, `truck`, or `bus`), examine a small region at the bottom-center of its bounding box. This region approximates where the object touches the road.

```text
Object footpoint overlaps drivable mask
-> object is a lead-object candidate

Object footpoint does not overlap drivable mask
-> exclude it from lead-object selection
```

Pedestrians, bicycles, and motorcycles remain in the fused data, but the first cruise-control baseline selects its lead object only from vehicle classes. This keeps the initial policy simple and clear. A later safety layer can handle vulnerable road users separately.

## Step 6B: Select the Lead Object

Among drivable vehicle candidates, select the one with the smallest radar distance. If no candidate passes the road-overlap test, select no lead object.

This means an object can be visible and radar-matched without becoming the lead object. For example, a parked car at the edge of the camera image should not be treated as an obstacle directly in the ego vehicle's path.

## Step 6C: Build Scene State

Create one JSON file with fixed keys. Missing numeric values remain `null`, not fabricated measurements.

```json
{
  "sample_index": 95,
  "lead_present": true,
  "lead_class": "bus",
  "lead_distance_m": 31.88,
  "lead_relative_speed_mps": 1.84,
  "lead_confidence": 0.7159,
  "lead_radar_point_count": 12,
  "drivable_ahead": true,
  "ego_speed_mps": 13.9
}
```

`ego_speed_mps` is a configurable fixed value for this offline baseline. nuScenes does not provide a direct speed measurement in the selected fused-obstacle file. The first script will use $13.9\,m/s$ (approximately $50\,km/h$) and record that assumption in the output.

`relative_speed_mps` is the radar compensated forward-axis velocity already produced by Phase 5. In this first baseline, a positive value is treated as a closing-risk signal and a value less than or equal to zero as non-closing. This is an approximation, not a true radial closing-speed calculation.

## Step 6D: Calculate Risk Values

When a lead object is present, calculate time to collision (TTC):

$$
TTC = \frac{distance}{relative\_speed}
$$

Calculate TTC only when `relative_speed_mps > 0`. Otherwise, record `ttc_s` as `null` because the simple controller does not predict a collision from a non-closing object.

## Step 6E: Apply the Baseline Rules

Use these transparent initial thresholds:

```text
No lead object and drivable area ahead
-> ACCELERATE

Lead distance <= 12 m
-> BRAKE

Lead is closing and TTC <= 4 s
-> BRAKE

Lead distance <= 30 m
-> MAINTAIN

No drivable area ahead
-> MAINTAIN

All other cases
-> ACCELERATE
```

The decision JSON must include a human-readable `decision_reason` so every recommendation is traceable to a specific rule.

## Outputs

```text
outputs/decision/
├── cam_front_sample_095_scene_state.json
├── cam_front_sample_095_rule_decision.json
└── cam_front_sample_095_decision_overlay.png
```

Example decision output:

```json
{
  "sample_index": 95,
  "recommended_action": "MAINTAIN",
  "decision_reason": "Lead bus is 31.88 m ahead and does not meet the brake threshold.",
  "lead_distance_m": 31.88,
  "lead_relative_speed_mps": 1.84,
  "ttc_s": 17.33,
  "ego_speed_mps": 13.9
}
```

The overlay will show the selected lead box, the drivable-area context, and the recommended action.

## Completion Criteria

```text
Phase 5 fused obstacles + Phase 3 drivable mask
-> one selected lead object or an explicit no-lead result
-> saved scene-state JSON
-> saved decision JSON with action and reason
-> decision overlay
```

## Connection to Future RL

The numerical scene state is intentionally compact so it can later become the observation for a PPO adaptive-cruise-control agent:

```text
[lead_present, lead_distance_m, lead_relative_speed_mps,
 drivable_ahead, ego_speed_mps]
```

PPO must run in a separate interactive simulation environment where actions change ego speed and relative distance. The Phase 6 rule-based controller is the baseline used to compare that future agent against transparent behavior.

## Multi-Sample Evaluation

Run the complete Phase 1-6 pipeline for a small, fixed set of samples and collect one comparable row per sample:

```bash
python phase6_multi_sample_evaluation.py --sample-indices 0 50 95
```

The evaluator reruns each phase for every requested index, then creates:

```text
outputs/evaluation/
├── phase1_to_6_evaluation.csv
└── phase1_to_6_evaluation.md
```

Each row reports the number of camera detections, the number that received a radar match, the selected lead vehicle, its distance, its TTC when closing, and the recommended action. A blank TTC means the selected lead's compensated radar velocity was not positive, so the first controller did not consider it a closing collision risk.