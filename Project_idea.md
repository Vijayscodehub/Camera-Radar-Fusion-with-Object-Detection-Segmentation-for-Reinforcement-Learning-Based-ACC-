# Camera-Radar Fusion for Obstacle Detection and Drivable Area Understanding

## Overview

This project implements a complete CPU-friendly perception and decision-making pipeline for autonomous driving research using the **nuScenes dataset**.

The system answers the following questions:

- What objects are around me?
- Where can I drive?
- How far away are obstacles?
- How fast are they moving?
- Should I brake, maintain speed, or accelerate?

The project combines:

- Camera-based object detection
- Semantic segmentation
- Drivable area understanding
- Radar-based distance and velocity estimation
- Camera-radar sensor fusion
- Reinforcement Learning for Adaptive Cruise Control (ACC)

The entire pipeline is designed to run on **CPU only** and intentionally avoids:

- ROS
- CARLA
- BEV representations
- LiDAR processing
- 3D object detection

---

## Project Goals

Build an obstacle-aware perception system capable of:

1. Detecting nearby objects.
2. Understanding drivable road regions.
3. Estimating obstacle distance.
4. Estimating relative obstacle velocity.
5. Generating driving decisions:
   - Brake
   - Maintain
   - Accelerate

---

## Constraints

✅ nuScenes Dataset

✅ CPU-Friendly

✅ No GPU Required

✅ No ROS

✅ No CARLA

✅ No BEV

✅ RL Extension Included

---

## Dataset

### Primary Dataset

**nuScenes**

### Sensors Used

- Camera Images
- Radar Data
- Bounding Box Annotations
- Vehicle Pose Information

### Ignored Components

- LiDAR
- BEV Tasks
- Tracking Challenge
- 3D Detection
- Multi-Sensor 3D Perception

### Focus

Only:

- Camera
- Radar

---

# Project Pipeline

```text
Camera
   ↓
YOLOv8n
   ↓
Detected Objects

Camera
   ↓
SegFormer-B0
   ↓
Semantic Segmentation
   ↓
Drivable Area

Radar
   ↓
Distance + Velocity

Object Detection + Radar
   ↓
Camera-Radar Fusion
   ↓
Obstacle-Aware Scene

Fusion + Segmentation
   ↓
State Generator
   ↓
PPO Agent
   ↓
Brake / Maintain / Accelerate
```

---

# Phase 0: Setup & Dataset Exploration

## Learning Objectives

- Python
- NumPy
- Pandas
- OpenCV
- PyTorch Basics

## Tasks

Create scripts or notebooks that:

- Load camera images
- Load radar frames
- Read annotations
- Visualize bounding boxes
- Explore dataset structure

## Deliverable

Visualization:

```text
Image
 + Bounding Boxes
 + Labels
```

Example:

```text
[CAR]
[PERSON]
[TRUCK]
```

### Skills Gained

- Dataset Handling
- OpenCV Visualization
- Annotation Parsing
- Sensor Data Inspection

---

# Phase 1: Object Detection

## Goal

Detect road participants from camera images.

### Classes

- Car
- Truck
- Bus
- Pedestrian
- Cyclist

## Model

### YOLOv8n

Use a pretrained model.

## Input

```text
Camera Image
```

## Output

```python
[
    {
        "class": "car",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95
    }
]
```

## Deliverable

Annotated image with detections:

```text
[CAR]
[PERSON]
[TRUCK]
```

### Skills Gained

- Object Detection
- Bounding Boxes
- OpenCV Drawing
- PyTorch Inference

---

# Phase 2: Semantic Segmentation

## Goal

Understand scene semantics.

## Model

### SegFormer-B0

Pretrained Hugging Face model.

## Classes

- Road
- Sidewalk
- Vehicle
- Pedestrian
- Building
- Vegetation
- Sky

## Input

```text
Camera Image
```

## Output

Colored segmentation mask.

### Example Colors

```text
Green = Road
Red   = Vehicle
Blue  = Sky
```

## Deliverable

```text
Original Image
+
Segmentation Overlay
```

### Skills Gained

- Semantic Segmentation
- Transformers
- Hugging Face
- Scene Parsing

---

# Phase 3: Drivable Area Extraction

## Goal

Convert segmentation results into a drivable map.

## Logic

```text
Road           → Drivable

Everything Else → Non-Drivable
```

## Output

Binary mask:

```text
1 = Drivable
0 = Non-Drivable
```

## Deliverable

Visualization:

```text
Safe Area
Unsafe Area
```

### Skills Gained

- Road Understanding
- Scene Understanding
- Binary Mask Generation

---

# Phase 4: Radar Processing

## Goal

Extract obstacle distance and velocity from radar data.

## Input

```text
Radar Frame
```

## Output

```python
{
    "distance": 24.5,
    "velocity": -3.2
}
```

## Interpretation

```text
Obstacle Distance : 24.5 m

Relative Velocity : -3.2 m/s
```

Negative velocity indicates the obstacle is approaching.

## Deliverable

Radar target visualization.

### Skills Gained

- Automotive Radar Basics
- Coordinate Systems
- Radar Data Processing

---

# Phase 5: Camera-Radar Fusion

## Goal

Associate camera detections with radar measurements.

### Example

Camera:

```text
Car detected
```

Radar:

```text
24 m
-3 m/s
```

Fusion Result:

```python
{
    "class": "car",
    "distance": 24,
    "relative_speed": -3
}
```

## Scene Summary Example

```text
Car Ahead

Distance: 24 m

Relative Speed: -3 m/s
```

## Deliverable

Obstacle-aware scene representation.

### Skills Gained

- Sensor Fusion
- Multimodal Perception
- Data Association

---

# Phase 6: Scene State Generator

## Goal

Convert perception outputs into RL-friendly numerical states.

## Inputs

- Object Detection
- Semantic Segmentation
- Radar Processing

## Output

```python
state = [
    distance_front,
    relative_speed,
    ego_speed,
    drivable_available
]
```

### Example

```python
[24, -3, 50, 1]
```

Meaning:

```text
Obstacle Distance = 24 m

Relative Speed = -3 m/s

Ego Speed = 50 km/h

Drivable Area Available = Yes
```

## Deliverable

State vector generator module.

---

# Phase 7: Reinforcement Learning Extension (ACC)

## Goal

Build a simple Adaptive Cruise Control (ACC) agent.

No simulator is required.

No CARLA is required.

## Environment

Custom Gymnasium Environment.

## State Space

```python
[
    distance_front,
    relative_speed,
    ego_speed
]
```

## Action Space

```python
0 = Brake

1 = Maintain

2 = Accelerate
```

## Algorithm

### PPO (Proximal Policy Optimization)

Using:

```text
Stable-Baselines3
```

## Reward Design

### Positive Rewards

- Safe following distance
- Smooth driving behavior

### Negative Rewards

- Collision
- Unsafe following distance
- Excessive braking
- Aggressive acceleration

## Desired Behavior

```text
Obstacle Close
    → Brake

Safe Distance
    → Maintain

Road Clear
    → Accelerate
```

## Deliverable

Trained PPO-based ACC agent.

### Skills Gained

- Reinforcement Learning
- PPO
- Reward Engineering
- Decision Making

---

# Suggested Project Structure

```text
camera_radar_fusion/
│
├── data/
│   ├── nuscenes/
│
├── notebooks/
│   ├── phase0_dataset_exploration.ipynb
│
├── src/
│   ├── detection/
│   │   └── yolo_detector.py
│   │
│   ├── segmentation/
│   │   └── segformer_segmenter.py
│   │
│   ├── drivable_area/
│   │   └── drivable_mask.py
│   │
│   ├── radar/
│   │   └── radar_processor.py
│   │
│   ├── fusion/
│   │   └── camera_radar_fusion.py
│   │
│   ├── state_generator/
│   │   └── state_builder.py
│   │
│   └── rl/
│       ├── acc_env.py
│       └── train_ppo.py
│
├── outputs/
│   ├── detections/
│   ├── segmentation/
│   ├── drivable_area/
│   ├── fusion/
│   └── rl_results/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Final Output Example

```python
{
    "object": "car",
    "distance": 24,
    "relative_speed": -3,
    "drivable": True,
    "recommended_action": "BRAKE"
}
```

---

# Skills Gained

## Strong Coverage

✅ Object Detection

✅ Semantic Segmentation

✅ Computer Vision

✅ OpenCV

✅ PyTorch

✅ Hugging Face

✅ Transformers

✅ NumPy

✅ Pandas

✅ Sensor Fusion

✅ Bounding Boxes

✅ Reinforcement Learning

✅ PPO

✅ State Representation

✅ Autonomous Driving Fundamentals

---

# Not Included (Intentionally)

❌ ROS

❌ CARLA

❌ BEV

❌ LiDAR

❌ 3D Detection

❌ End-to-End Autonomous Driving

---

## Conclusion

This project provides a complete end-to-end perception and decision-making pipeline using only **Camera + Radar** data from the **nuScenes dataset**. It progresses from basic perception tasks to practical sensor fusion and finally to reinforcement learning-based adaptive cruise control, while remaining lightweight, CPU-friendly, and highly educational.

**Mini**![minus](https://www.nuscenes.org/static/media/up-arrow.e0930d46.svg)

Subset of trainval, 10 scenes, used to explore the data without downloading the whole dataset.

![download](https://www.nuscenes.org/static/media/download.c899fbad.svg)Metadata and sensor file blobs
