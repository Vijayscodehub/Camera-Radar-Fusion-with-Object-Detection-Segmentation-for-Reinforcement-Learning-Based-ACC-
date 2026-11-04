## Instead, Phase 5 can simply become:

```text
1. Run YOLO on camera image.

2. Read radar points from the corresponding radar sample.

3. Transform radar points into the camera frame
using nuScenes calibration records.

4. Project radar detections onto the image.

5. Associate radar points with YOLO bounding boxes.

6. Produce:
class
distance
relative speed
```

That's a much more realistic CPU-friendly scope.

<br><br>

## Honestly, I'd simplify your project further

For a first version, you could completely avoid coordinate-transform math and do:

```text
Phase 5A
--------
Use nearest radar target
for each front-camera detection.

Output:
Car
Distance = 22 m
Speed = -2.5 m/s
```

Then later:

```text
Phase 5B
--------
Radar-to-camera projection
using nuScenes calibration
```