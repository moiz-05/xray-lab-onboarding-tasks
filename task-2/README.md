# Fruit Detection with MMDetection

This task trains an RTMDet object detection model on a small fruit dataset using
MMDetection. The dataset uses COCO-style annotations and contains four classes:

- avocado
- apple
- banana
- grape

The main training entry point is `training/train.py`. It loads the RTMDet Tiny
configuration from the local `mmdetection/` checkout, rewrites the dataset paths
for this project, trains for 2 epochs on CPU, and writes checkpoints to
`outputs/`.

## Project Structure

```text
fruit-mmdetection/
  configs/                  Custom config experiments
  dataset/
    _annotations.coco.json  COCO annotation file
    train/                  Training images
  inference/
    test_inference.py       Simple inference/visualization script
  training/
    train.py                Main training script
  check.py                  Verifies OpenMMLab imports
  requirements.txt          Working dependency pins
```

Generated folders such as `venv/`, `outputs/`, `checkpoints/`, and the cloned
`mmdetection/` source are ignored by Git.

## Dependency Notes

There were some difficulties caused by dependency version mismatches.

The first visible issue was:

```text
Import "mmcv" could not be resolved
```

The root causes were:

- `mmcv` was not installed in the active virtual environment.
- `mmcv-lite` could import as `mmcv`, but MMDetection needed compiled
  `mmcv.ops`, so full `mmcv==2.1.0` was required.
- The environment originally had a newer PyTorch version that did not have a
  matching prebuilt MMCV wheel, so MMCV tried to compile from source on Windows.
- NumPy 2.x was incompatible with the PyTorch/MMCV wheel combination, so NumPy
  had to be pinned below 2.
- The newest OpenCV package required NumPy 2.x, so OpenCV was pinned to a
  NumPy 1.x-compatible version.

The working dependency set is pinned in `requirements.txt`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install PyTorch CPU wheels:

```powershell
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the rest of the dependencies:

```powershell
python -m pip install -r requirements.txt
python -m mim install mmcv==2.1.0
```

If VS Code still reports missing imports, select this interpreter:

```text
.\venv\Scripts\python.exe
```

## Verify Installation

Run:

```powershell
python check.py
```

Expected output should include versions for `mmcv`, `mmengine`, and `mmdet`.

Example:

```text
2.1.0
0.10.7
3.3.0
```

## Training

Run training from the project root:

```powershell
python training\train.py
```

The script currently:

- trains RTMDet Tiny on CPU
- uses `dataset/_annotations.coco.json`
- uses images from `dataset/train/`
- trains for 2 epochs
- saves outputs to `outputs/`
- disables the pretrained backbone download so training can run without network
  access

After training, a checkpoint is saved, for example:

```text
outputs/epoch_2.pth
```

The run also creates MMEngine log folders inside `outputs/`.

## Inference

The inference script is:

```powershell
python inference\test_inference.py
```

Before running it, make sure these paths in `inference/test_inference.py` match
your actual files:

```python
CONFIG = '../configs/fruits_rtmdet_config.py'
CHECKPOINT = '../outputs/latest.pth'
IMAGE = '../dataset/train/sample.jpg'
```

If training produced `outputs/epoch_2.pth`, update `CHECKPOINT` to:

```python
CHECKPOINT = '../outputs/epoch_2.pth'
```

Also update `IMAGE` to point to an existing image in `dataset/train/`.

## Notes

This task was tested on Windows with CPU-only training. Training is slow on CPU,
so a short 2-epoch run is used mainly to verify that the full pipeline works.
For better model quality, train for more epochs and preferably use a GPU-enabled
PyTorch/MMCV environment.
