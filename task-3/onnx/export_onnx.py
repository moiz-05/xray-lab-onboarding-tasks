import os
import sys
import torch

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

# =========================================================
# IMPORTS
# =========================================================

from mmengine.config import Config

from mmdet.apis import init_detector

# =========================================================
# CONFIGURATION
# =========================================================

CONFIG_FILE = os.path.join(
    PROJECT_ROOT,
    "configs",
    "fruits_rtmdet_config.py"
)

CHECKPOINT_FILE = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "latest.pth"
)

ONNX_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "onnx",
    "models",
    "rtmdet_fruits.onnx"
)

# =========================================================
# EXPORT SETTINGS
# =========================================================

INPUT_HEIGHT = 640
INPUT_WIDTH = 640

OPSET_VERSION = 11

DYNAMIC_AXES = True

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    os.path.dirname(ONNX_OUTPUT_PATH),
    exist_ok=True
)

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Loading RTMDet Model")
print("=" * 60)

model = init_detector(
    CONFIG_FILE,
    CHECKPOINT_FILE,
    device="cpu"
)

model.eval()

print("Model Loaded Successfully")

# =========================================================
# DUMMY INPUT
# =========================================================

dummy_input = torch.randn(
    1,
    3,
    INPUT_HEIGHT,
    INPUT_WIDTH
)

# =========================================================
# DYNAMIC AXES
# =========================================================

dynamic_axes = None

if DYNAMIC_AXES:

    dynamic_axes = {
        "input": {
            0: "batch_size",
            2: "height",
            3: "width"
        },
        "output": {
            0: "batch_size"
        }
    }

# =========================================================
# EXPORT ONNX
# =========================================================

print("\nExporting ONNX Model...")
print(f"Opset Version : {OPSET_VERSION}")
print(f"Input Shape   : {dummy_input.shape}")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes
)

print("\nONNX Export Successful")
print(f"Saved Model: {ONNX_OUTPUT_PATH}")

print("\n" + "=" * 60)
print("ONNX Export Complete")
print("=" * 60)