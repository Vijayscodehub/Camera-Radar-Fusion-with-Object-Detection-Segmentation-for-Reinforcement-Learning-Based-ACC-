# Camera-Radar Fusion for Obstacle Detection and Drivable Area Understanding

## Project Scope

This is a CPU-friendly autonomous-driving perception project using the nuScenes dataset.

The main project uses only camera and radar data to answer:

- What objects are in front of the vehicle?
- Where is the drivable road area?
- How far away is an obstacle?
- Is an obstacle moving closer or farther away?
- Should the vehicle brake, maintain speed, or accelerate?

The project intentionally excludes ROS, CARLA, LiDAR processing, BEV representations, and 3D object detection. Reinforcement Learning (RL) is an optional final extension, not a requirement for the initial project.

## Pipeline

```text
Camera -> YOLOv8n -> Object detections
Camera -> SegFormer-B0 -> Semantic segmentation -> Drivable-area mask
Radar -> Range, position, and relative velocity
Camera detections + radar -> Camera-radar fusion -> Obstacle-aware scene
Scene state -> Rule-based recommendation
Scene state -> PPO ACC agent (optional future extension)
```

## Why the Project Is Feasible

nuScenes provides the metadata needed to work with its sensors:

- Camera intrinsics.
- Camera and radar positions and rotations relative to the ego vehicle.
- Ego-vehicle pose for each sensor frame.
- Sensor timestamps.
- 3D object annotations for the training and validation data.

The task is not to calibrate sensors from scratch. The important implementation task is correctly using nuScenes calibration and pose data to project radar points onto the front-camera image.

Camera, radar, and annotation timestamps are close but may not be identical. The initial implementation should use the radar sample closest to each `CAM_FRONT` frame.

## Implementation Plan

### Phase 0: Dataset Exploration

Use the nuScenes mini split first. Load a front-camera image, the closest front-radar sample, and its annotations.

Verify the image dimensions, timestamps, camera intrinsics, sensor extrinsics, ego pose, and coordinate conventions. Visualize annotated 3D boxes projected into the camera image.

### Phase 1: Camera Object Detection

Run pretrained YOLOv8n on `CAM_FRONT`. Keep the road-user categories relevant to the project: car, truck, bus, pedestrian, bicycle, and motorcycle.

Save structured detections as well as annotated images:

```python
{
    "class": "car",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.95
}
```

### Phase 2: Semantic Segmentation

Run a Cityscapes-finetuned SegFormer-B0 model to obtain a semantic image mask. This provides explicit road information and is more useful than a generic segmentation model for drivable-area extraction.

### Phase 3: Drivable-Area Extraction

Convert the segmentation road class into a binary image-aligned mask:

```text
Road -> drivable (1)
Everything else -> non-drivable (0)
```

Use the mask as supporting context. It helps decide whether a detected object is likely in the vehicle's path, but it should not be the only source of a driving decision.

### Phase 4: Radar Processing

Read `RADAR_FRONT` points and retain their Cartesian position, range, and compensated radial velocity. Apply validity filters, but retain raw values during early debugging so filtering behavior remains visible.

Radar can contain noise, roadside returns, multiple points from one vehicle, or no point for a visible vehicle. A missing radar match is a valid outcome.

### Phase 5A: Radar-to-Camera Projection

Transform radar points through the coordinate frames supplied by nuScenes:

```text
Radar coordinates -> ego-vehicle coordinates -> camera coordinates -> image pixels
```

Draw the resulting image points over the camera image. This is a separate milestone because it proves the calibration, pose, and transformation code is correct before attempting object association.

### Phase 5B: Camera-Radar Association

Match a projected radar point to a YOLO detection only when the point falls inside that detection's bounding box. For boxes with several points, use a robust statistic such as median range and median relative velocity.

Example fusion result:

```python
{
    "class": "car",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.92,
    "distance_m": 24.1,
    "relative_speed_mps": -2.7,
    "radar_point_count": 3,
    "radar_matched": True
}
```

The main difficulty of the project is this association step. A radar point inside a box is a useful initial rule, not proof of a perfect match, especially for overlapping vehicles.

### Phase 6: Scene State and Rule-Based Baseline

For cruise-control decisions, select a lead object rather than treating every detected object as equally important. Initially, choose the nearest radar-matched object that is near the center of the front camera and overlaps the drivable area.

Create a stable scene state:

```python
{
    "lead_present": True,
    "lead_distance_m": 24.1,
    "lead_relative_speed_mps": -2.7,
    "ego_speed_mps": 13.9,
    "drivable_ahead": True,
    "lead_confidence": 0.92
}
```

First use transparent rules:

```text
Lead object too close or time-to-collision too low -> BRAKE
Lead object safely ahead -> MAINTAIN
No lead object -> ACCELERATE
```

This controller is the baseline for testing the perception pipeline and for evaluating any later RL agent.

### Phase 7: Optional RL ACC Extension

PPO can use a numerical form of the Phase 6 state. It should run in a separate custom Gymnasium environment, where ego and lead-vehicle positions respond to each action.

nuScenes recordings are useful for realistic state ranges and perception noise, but cannot by themselves train an action-based RL controller because recorded vehicles do not react to the agent's brake or acceleration decisions.

Compare PPO against the rule-based controller using collision count, unsafe following time, speed progress, and control smoothness.

## Evaluation

Evaluate each layer separately rather than only showing final screenshots:

- Detection: whether YOLO detects relevant annotated objects.
- Projection: visual correctness of radar points on camera images.
- Association: whether radar points map to the correct annotated object.
- Range: difference between fused radar range and the annotated object distance.
- Velocity: whether estimated relative velocity agrees with objects approaching or moving away across frames.
- Decision: whether the baseline reacts sensibly in close, safe, and clear-road cases.

For example, an annotated vehicle $25\,m$ ahead and a fused estimate of $24\,m$ has an absolute range error of $1\,m$.

## Main Risks

The highest-risk area is camera-radar association, followed by radar-to-camera projection. Segmentation quality can be imperfect in shadows, construction zones, or uncommon road layouts, so use it as context rather than absolute truth.

CPU inference is practical for offline processing. Process a small set of scenes first, resize inputs appropriately, use frame skipping when needed, and save intermediate results so models do not need to rerun during fusion debugging.

## Expected Final Output

```python
{
    "object": "car",
    "distance_m": 24.0,
    "relative_speed_mps": -3.0,
    "drivable_ahead": True,
    "recommended_action": "BRAKE"
}
```