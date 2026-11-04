# Phase 5: Camera-Radar Fusion

## Goal

Combine camera object detections with radar distance and velocity measurements.

```text
YOLO detection: "bus"
+ radar target: distance and velocity
-> fused obstacle: class + distance + relative speed
```

Phase 5 has two separate parts:

```text
Phase 5A: Project radar targets onto the camera image.
Phase 5B: Associate projected radar targets with YOLO bounding boxes.
```

Splitting the work makes debugging possible. If a final match is wrong, we can first check whether the radar point was projected into the wrong image position or whether a correctly projected point was associated with the wrong detection.

## Inputs

Use the same nuScenes sample index across all phases.

```text
Phase 1
outputs/detections/cam_front_sample_095_yolo11n.json

Phase 4
outputs/radar/radar_front_sample_095_targets.json

nuScenes metadata
calibrated_sensor.json
+ ego_pose.json
+ sample_data.json

nuScenes sensor data
samples/CAM_FRONT/
+ samples/RADAR_FRONT/
```

## Phase 5A: Radar-to-Camera Projection

### Goal

Convert each retained radar target from radar coordinates into a pixel location on the corresponding front-camera image.

```text
Radar point (x, y, z)
-> radar sensor frame
-> ego-vehicle frame
-> global frame
-> camera ego-vehicle frame
-> camera frame
-> image pixel (u, v)
```

### Why Ego Pose Is Needed

The radar and camera do not capture at exactly the same time. The ego vehicle may move slightly between their timestamps. Using each sensor frame's ego pose lets us compensate for that movement before projecting radar into the camera.

### Projection Rules

Keep a projected point only when it is:

```text
- in front of the camera
- inside the image width and height
- a valid filtered Phase 4 target
```

### Phase 5A Outputs

```text
outputs/fusion/
├── cam_front_sample_095_radar_projection.png
└── cam_front_sample_095_projected_radar_targets.json
```

The PNG will show the original camera image with projected radar points. The JSON will preserve each radar target's image-pixel position, distance, and velocity.

Example projected target:

```json
{
  "radar_point_index": 1,
  "pixel_x": 742,
  "pixel_y": 493,
  "distance_m": 24.53,
  "relative_velocity_mps": -3.2
}
```

### Phase 5A Completion Criteria

```text
Filtered RADAR_FRONT targets
-> calibration and pose transformations
-> projected radar points on CAM_FRONT image
-> projection PNG and JSON
```

## Phase 5B: Camera-Radar Association

### Goal

Connect a projected radar target to a YOLO detection.

Initial association rule:

```text
Projected radar point falls inside a YOLO bounding box
-> point is a candidate match for that detection
```

A box can contain several radar points. For the first version, use the median target distance and median velocity for all points inside that box. A detection with no point inside its box must remain `radar_matched: false`.

### Fused Output

```json
{
  "class": "bus",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.92,
  "distance_m": 24.1,
  "relative_speed_mps": -2.7,
  "radar_point_count": 3,
  "radar_matched": true
}
```

### Important Limitation

A radar point inside a detection box is a useful first association rule, not proof that it belongs to that object. Nearby, overlapping, or occluded objects can produce ambiguous matches. The project should preserve the point count and unmatched status so this uncertainty stays visible.

### Phase 5B Outputs

```text
outputs/fusion/
├── cam_front_sample_095_fused_obstacles.json
└── cam_front_sample_095_fusion_overlay.png
```

The overlay will show YOLO boxes and their associated radar distance and velocity values.

## Why This Phase Matters

Camera tells the project what an object likely is. Radar tells the project how far it is and how its relative position changes.

```text
Camera: car
Radar: 24 m, approaching
Fusion: car at 24 m, approaching
```

This obstacle-aware output is the input for the later lead-object selection, rule-based decision system, and optional RL ACC extension.
