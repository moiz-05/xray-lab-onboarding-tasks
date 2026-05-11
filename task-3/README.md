# RTMDet Fruit Detection Project

Object detection project using RTMDet (Real-Time Detector) for fruit classification on four classes: avocado, apple, banana, and grape.

## Project Structure

```
task-3/
├── configs/                  # MMDetection config files
│   ├── fruits_rtmdet_config.py    # Main config (custom classes)
│   ├── rtmdet_tiny_8xb32-300e_coco.py
│   └── rtmdet_s_8xb32-300e_coco.py
├── checkpoints/              # Trained model weights
│   └── latest.pth
├── dataset/                  # COCO format dataset
├── inference/                # Inference scripts & outputs
│   ├── images/               # Input images
│   ├── outputs/
│   │   ├── mmdet/           # MMDetection results
│   │   └── onnx/            # ONNX inference results
│   ├── mmdet_inference.py   # MMDetection inference
│   └── onnx_inference.py    # ONNX Runtime inference
├── onnx/                     # ONNX model files
│   └── models/
│       └── rtmdet_fruits.onnx
├── utils/                    # Utility functions
│   ├── preprocessing.py     # Image loading utilities
│   └── visualization.py      # Visualization helpers
└── requirements.txt          # Python dependencies
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Classes

The model detects four fruit classes:
- avocado
- apple
- banana
- grape

## Inference

### MMDetection Inference

Run inference using the trained MMDetection model:

```bash
python inference/mmdet_inference.py
```

Output images are saved to `inference/outputs/mmdet/`

### ONNX Inference

Run inference using the exported ONNX model:

```bash
python inference/onnx_inference.py
```

Output images are saved to `inference/outputs/onnx/`

## Training

To train the model, use MMDetection's training tools:

```bash
mim train mmdet configs/fruits_rtmdet_config.py
```

## Model Export

To export the trained model to ONNX format:

```bash
python tools/deploy.py \
    configs/fruits_rtmdet_config.py \
    checkpoints/latest.pth \
    --output-dir onnx/models/ \
    --format onnx
```

## Configuration

The main config `configs/fruits_rtmdet_config.py` inherits from:
- `rtmdet_tiny_8xb32-300e_coco.py`
- `rtmdet_s_8xb32-300e_coco.py`

Key settings:
- Input size: 640x640
- Number of classes: 4
- Dataset: COCO format

## Dependencies

- Python 3.x
- PyTorch 2.1+
- MMDetection 3.3+
- MMEngine 0.10+
- MMCV 2.1+
- OpenCV
- ONNX Runtime
- NumPy