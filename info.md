Detection success: did YOLO detect the annotated vehicle?
Association success: did the radar point get assigned to the correct vehicle?
Range error: how close is estimated range to annotated range?
Velocity plausibility: does the radar radial velocity agree with the object getting closer or farther across frames?



Project Phases

Setup and Dataset Check

Install Python packages and nuscenes-devkit.
Download/use v1.0-mini or v1.0-trainval.
Load one CAM_FRONT image, one RADAR_FRONT frame, and annotations.
Verify calibration, timestamps, and file paths.
Object Detection

Run YOLOv8n on front-camera images.
Detect cars, trucks, buses, pedestrians, bicycles, and motorcycles.
Save bounding boxes, labels, and confidence scores.
Semantic Segmentation

Run SegFormer-B0 on the same front-camera image.
Produce pixel labels such as road, sidewalk, vehicle, sky, and building.
Drivable Area Extraction

Convert road pixels from Phase 2 into a binary mask.
1 = drivable, 0 = non-drivable.
Overlay the mask on the original image.
Radar Processing

Load front-radar points.
Extract each point’s position, distance, and relative velocity.
Filter unusable/noisy radar points.
Camera-Radar Fusion

5A: Projection: transform radar points into camera-image pixels.
5B: Association: match projected radar points with YOLO bounding boxes.
Output an obstacle list with object class, distance, and relative speed.
Scene State and Rule-Based Decisions

Select the likely lead object ahead.
Create a stable state: distance, relative speed, ego speed, and drivable area.
Apply simple rules for BRAKE, MAINTAIN, or ACCELERATE.
Optional RL ACC Extension

Build a small custom Gymnasium environment.
Train PPO using the Phase 6 state format.
Compare PPO decisions against the rule-based baseline.
Each phase should produce a visible output before moving on. For example, Phase 1 produces an annotated image; Phase 5A produces camera images with projected radar points.

We should start with Phase 0: setup and loading one camera-radar sample