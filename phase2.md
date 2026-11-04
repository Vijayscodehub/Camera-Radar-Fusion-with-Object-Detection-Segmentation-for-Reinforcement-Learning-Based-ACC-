Yes, Phase 1 is achieved: you ran YOLO on several camera samples, created annotated images and JSON detections, and committed it.

Phase 2 Layout: Semantic Segmentation

Goal: label every pixel in the same CAM_FRONT image.
Input:
CAM_FRONT image

Process:
Pretrained SegFormer-B0 on CPU

Output:
Every pixel gets a class:
road, sidewalk, car, person, building, sky, vegetation, etc.

For example:
Original image              Segmentation result
----------------            -------------------
road area                   green pixels
sky                         blue pixels
cars                        red pixels
buildings                   gray pixels

What we will implement

Install Hugging Face transformers.
Download a pretrained Cityscapes SegFormer-B0 model.
Give it a chosen CAM_FRONT image, for example sample 95.
Create:
a raw class-ID mask, where each pixel is a number
a colored segmentation image
an overlay combining the original image and colored mask

Save outputs under:
outputs/segmentation/


Why semantic segmentation?

YOLO gives boxes around objects:
car box
person box

SegFormer gives areas:

all road pixels
all sidewalk pixels
all sky pixels


Then Phase 3 takes only the road pixels and creates the simple drivable mask:
road -> 1
everything else -> 0

Phase 2 complete when
CAM_FRONT image
-> SegFormer-B0
-> colored semantic mask
-> overlay image

We will run it explicitly on CPU, just like YOLO.