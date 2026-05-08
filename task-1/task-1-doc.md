# Dataset Annotation Task Documentation

## Objective
The objective of this task was to create an annotated fruit dataset for object detection and instance segmentation using Roboflow, then export it in COCO JSON format for future model training and experimentation.

---

# Tools Used

- Annotation platform: Roboflow
- Export format: COCO JSON / COCO MMDetection
- Annotation types:
  - Bounding box annotation for object detection
  - Polygon annotation for instance segmentation

---

# Dataset Overview

The exported training set contains:

- 192 image files
- 196 object annotations
- 4 fruit classes
- Image size: 512 x 512 pixels for the exported images
- Annotation file: `_annotations.coco.json`

The dataset contains fruit images with the following classes:

| Category ID | Class Name |
|-------------|------------|
| 1 | avocado |
| 2 | apple |
| 3 | banana |
| 4 | grape |

---

# Annotation Tasks Performed

## 1. Object Detection Annotation

Bounding box annotations are included for each labeled object in the dataset.

### COCO Fields Used

- `image_id`: links the annotation to an image entry
- `category_id`: links the object to one of the fruit classes
- `bbox`: stores the bounding box as `[x, y, width, height]`
- `area`: stores the annotation area

### Example

```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 1,
  "bbox": [102, 110, 310, 304],
  "area": 94240
}
```

---

## 2. Instance Segmentation Annotation

Instance segmentation masks are stored in `_annotations.coco.json` as COCO polygon coordinates. They are not exported as separate PNG mask images in this dataset format.

Each `segmentation` field contains one or more polygons. A polygon is represented as a flat list of x/y coordinate pairs:

```text
[x1, y1, x2, y2, x3, y3, ...]
```

### Example

```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 1,
  "bbox": [102, 110, 310, 304],
  "segmentation": [
    [
      102, 267,
      106, 317,
      117, 345
    ]
  ],
  "iscrowd": 0
}
```

The `height` and `width` values in the `images` section describe the image dimensions only. They are required metadata, but they are not enough for segmentation. The actual segmentation mask information is stored in the `segmentation` polygons.

---

# Label Taxonomy

The following taxonomy was used to ensure annotation consistency:

| Class Name |
|------------|
| avocado |
| apple |
| banana |
| grape |

### Naming Rules

- Lowercase labels only
- Singular nouns only
- Consistent naming throughout the dataset

---

# Quality Assurance (QA)

A basic QA pass was performed to validate annotation quality.

## QA Checks Performed

- Verified label consistency across the four classes
- Checked that annotations are connected to valid image IDs
- Ensured bounding boxes are present for object detection
- Confirmed segmentation polygons are present for instance segmentation
- Confirmed `iscrowd` is set to `0` for polygon instance annotations

---

# Dataset Export

The annotated dataset was exported in COCO JSON format from Roboflow.

## Exported Outputs

- Original/exported image files as `.jpg`
- COCO annotation file: `_annotations.coco.json`

No separate binary mask image files are included in this export. If PNG masks are required, the COCO polygon annotations can be rasterized into mask images using a conversion script.

---

# COCO JSON Structure

The exported JSON file contains:

- `info`: export metadata from Roboflow
- `licenses`: dataset license metadata
- `categories`: class IDs and class names
- `images`: image filenames, image IDs, height, and width
- `annotations`: object boxes, areas, segmentation polygons, image IDs, and category IDs

Example top-level components:

```json
{
  "info": {},
  "licenses": [],
  "categories": [],
  "images": [],
  "annotations": []
}
```
