# Phase 3: Drivable Area Extraction

## Goal

Convert the detailed Phase 2 semantic-segmentation result into one simple answer for every image pixel:

```text
Can the ego vehicle drive here?

Yes -> 1
No  -> 0
```

## Input

Use the class-ID mask created by Phase 2:

```text
outputs/segmentation/
└── cam_front_sample_095_segformer_b0_class_ids.png
```

This image contains one class number for every pixel. It is not a normal colored image.

The Cityscapes SegFormer model uses:

```text
Class ID 0 = road
```

## Processing Steps

```text
Phase 2 class-ID mask
-> find all pixels with class ID 0
-> set those road pixels to 1
-> set every other pixel to 0
-> create a binary drivable-area mask
```

For visualization, the binary mask will use:

```text
White = drivable road
Black = non-drivable
```

We can apply light image-cleaning operations if required:

```text
Small holes inside road area -> fill
Small isolated road pixels -> remove
```

These operations must be used carefully. Strong cleaning could incorrectly label a sidewalk, vehicle, or pedestrian area as drivable.

## Outputs

Save these files under:

```text
outputs/drivable_area/
```

Expected files:

```text
cam_front_sample_095_drivable_binary.png
cam_front_sample_095_drivable_overlay.png
cam_front_sample_095_drivable_stats.json
```

### Binary Mask

```text
White pixels = road / drivable
Black pixels = everything else / non-drivable
```

### Drivable-Area Overlay

The original `CAM_FRONT` image with drivable road pixels highlighted in green.

### Statistics JSON

Example:

```json
{
	"sample_index": 95,
	"road_class_id": 0,
	"drivable_pixels": 432100,
	"total_pixels": 1440000,
	"drivable_percentage": 30.01
}
```

## Why This Phase Matters

Phase 2 labels many scene types, such as road, vehicles, people, sky, and buildings. Phase 3 reduces that detailed output into a simple drivable-area map.

Later, the project will combine:

```text
Drivable area
+ YOLO object detections
+ Radar distance and relative velocity
-> likely lead object
-> brake / maintain / accelerate decision
```

## Completion Criteria

Phase 3 is complete when the project can produce:

```text
Phase 2 class-ID mask
-> road-only binary mask
-> green drivable-area overlay
-> statistics JSON
```
