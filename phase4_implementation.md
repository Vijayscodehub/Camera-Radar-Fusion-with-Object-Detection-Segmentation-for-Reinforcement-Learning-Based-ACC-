# Phase 4: Radar Processing

## Goal

Turn raw `RADAR_FRONT` point-cloud rows into a clean, understandable list of radar targets.

```text
RADAR_FRONT .pcd file
-> extract position and velocity
-> calculate distance
-> remove unusable points
-> save clean radar targets
```

## Input

Use the front-radar `.pcd` file linked to the selected nuScenes sample:

```text
samples/RADAR_FRONT/
```

The script will use the same `--sample-index` pattern as earlier phases. This ensures the radar data comes from the same moment as the camera image, YOLO detections, and segmentation outputs.

## Raw Radar Values

Each radar return contains several values. For the initial project, use:

```text
x             = forward distance from the radar in meters
 y            = left/right distance from the radar in meters
vx_comp       = ego-motion-compensated forward velocity in meters per second
vy_comp       = ego-motion-compensated left/right velocity in meters per second
```

Radar coordinate convention:

```text
x > 0  = in front of the ego vehicle
y > 0  = to the left of the ego vehicle
y < 0  = to the right of the ego vehicle
```

## Target Format

For every retained radar point, create a clean target object:

```python
{
    "x_m": 24.5,
    "y_m": -1.2,
    "distance_m": 24.53,
    "relative_velocity_mps": -3.2
}
```

Meaning:

```text
x_m                    = forward distance from the radar
 y_m                   = sideways distance from the radar
distance_m             = direct distance from the radar
relative_velocity_mps  = forward relative velocity after ego-motion compensation
```

For the initial version:

```text
Negative relative velocity -> object is getting closer
Positive relative velocity -> object is moving farther away
```

## Processing Steps

```text
1. Load the selected nuScenes sample.
2. Find its RADAR_FRONT file.
3. Read radar points with RadarPointCloud.
4. Extract x, y, vx_comp, and vy_comp.
5. Calculate distance:

   distance_m = sqrt(x_m^2 + y_m^2)

6. Filter unusable radar points.
7. Save retained targets as JSON.
8. Create a top-down visualization.
```

## Initial Filtering Rules

Keep only points that are:

```text
- In front of the vehicle: x_m > 0
- Within a useful distance range
- Marked valid by radar metadata
```

Suggested initial distance range:

```text
2 m to 100 m
```

Keep the filters simple at first. Radar can return noise, roadside objects, several points from one vehicle, or no point for a visible vehicle. A visible object with no radar point is a valid situation.

## Outputs

Save results under:

```text
outputs/radar/
```

Expected files:

```text
radar_front_sample_095_targets.json
radar_front_sample_095_top_down.png
```

### Targets JSON

A list of clean radar targets for later Phase 5 projection and association.

### Top-Down Visualization

```text
Black triangle = ego vehicle / radar position at (0, 0)
Each colored point = retained radar target
Right direction = forward from the vehicle
Up direction = left of the vehicle
Color = compensated forward velocity
```

## Why This Phase Matters

Camera detection tells us what an object looks like, but radar adds distance and relative velocity.

Later, Phase 5 will use:

```text
YOLO bounding box
+ cleaned radar target
-> project radar point onto camera image
-> match radar point to detection
-> object class + distance + relative speed
```

## Completion Criteria

Phase 4 is complete when the project can produce:

```text
RADAR_FRONT point cloud
-> filtered radar targets with position, distance, and relative velocity
-> targets JSON
-> top-down radar visualization
```

## Implementation Understanding

### What Phase 4 Does

Phase 4 turns raw radar measurements into cleaner data we can use later.

Raw radar gives many rows of numbers. For sample `95`, it contained:

```text
60 radar points
```

A point is only a reflection detected by radar. It could be from:

- A car
- A bus
- A road barrier
- A sign
- A building
- Noise

Radar does not tell us the object class. That is why YOLO is needed later.

### Problem We Faced

Raw radar data is not immediately understandable. Each point has 18 values:

```text
x, y, z, velocity, radar strength, validity flags, and more
```

We had to choose the useful values:

```text
x_m      -> how far forward the point is
y_m      -> how far left/right it is
distance -> direct distance from radar
vx_comp  -> forward velocity after removing ego-car motion
```

The main problem was deciding which points are trustworthy. Some points are invalid according to nuScenes radar metadata.

### How We Solved It

We used this filter:

```text
x_m > 0                  -> point is in front of the vehicle
2 m <= distance <= 100 m -> useful nearby range
invalid_state == 0       -> nuScenes marks the point as valid
```

This reduced:

```text
60 raw points
-> 55 usable radar targets
```

We did not remove points merely because they look unusual. At this stage, we want to preserve valid radar information for later camera-radar matching.

### Why We Saved JSON

The JSON gives a clean list like:

```json
{
    "radar_point_index": 1,
    "x_m": 7.0,
    "y_m": 9.1,
    "distance_m": 11.48,
    "relative_velocity_mps": -0.74,
    "lateral_velocity_mps": -0.97
}
```

Phase 5 will project this exact point onto the camera image. Then we can ask:

```text
Does this radar point land inside a YOLO car or bus box?
```

If yes, we can combine:

```text
YOLO: "bus"
Radar: "distance 24 m, velocity ..."
```

One small accuracy note: `relative_velocity_mps` currently uses the radar's compensated **forward** velocity (`vx_comp`). Later, for a more precise "approaching or moving away" value, we can calculate velocity along the direct line from radar to object, not only the forward axis.
