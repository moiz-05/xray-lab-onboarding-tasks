import os
import sys

print("=" * 60)
print("MMDetection + ONNX Environment Verification")
print("=" * 60)

# =========================================================
# PYTHON VERSION
# =========================================================

print("\n[1] Python Version")

print(sys.version)

# =========================================================
# TORCH
# =========================================================

print("\n[2] PyTorch")

try:
    import torch

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")

except Exception as e:
    print(f"PyTorch Error: {e}")

# =========================================================
# MMCV
# =========================================================

print("\n[3] MMCV")

try:
    import mmcv

    print(f"MMCV Version: {mmcv.__version__}")

except Exception as e:
    print(f"MMCV Error: {e}")

# =========================================================
# MMCV OPS
# =========================================================

print("\n[4] MMCV OPS")

try:
    from mmcv.ops import roi_align

    print("MMCV OPS Working")

except Exception as e:
    print(f"MMCV OPS Error: {e}")

# =========================================================
# MMENGINE
# =========================================================

print("\n[5] MMEngine")

try:
    import mmengine

    print(f"MMEngine Version: {mmengine.__version__}")

except Exception as e:
    print(f"MMEngine Error: {e}")

# =========================================================
# MMDETECTION
# =========================================================

print("\n[6] MMDetection")

try:
    import mmdet

    print(f"MMDetection Version: {mmdet.__version__}")

except Exception as e:
    print(f"MMDetection Error: {e}")

# =========================================================
# ONNX
# =========================================================

print("\n[7] ONNX")

try:
    import onnx

    print(f"ONNX Version: {onnx.__version__}")

except Exception as e:
    print(f"ONNX Error: {e}")

# =========================================================
# ONNX RUNTIME
# =========================================================

print("\n[8] ONNX Runtime")

try:
    import onnxruntime as ort

    print(f"ONNX Runtime Version: {ort.__version__}")

except Exception as e:
    print(f"ONNX Runtime Error: {e}")

# =========================================================
# CHECKPOINT VALIDATION
# =========================================================

print("\n[9] Checkpoint Validation")

checkpoint_path = "checkpoints/latest.pth"

if os.path.exists(checkpoint_path):
    print(f"Checkpoint Found: {checkpoint_path}")
else:
    print(f"Checkpoint Missing: {checkpoint_path}")

# =========================================================
# CONFIG VALIDATION
# =========================================================

print("\n[10] Config Validation")

config_path = "configs/rtmdet_fruits_config.py"

if os.path.exists(config_path):
    print(f"Config Found: {config_path}")
else:
    print(f"Config Missing: {config_path}")

# =========================================================
# DATASET VALIDATION
# =========================================================

print("\n[11] Dataset Validation")

dataset_path = "dataset/_annotations.coco.json"

if os.path.exists(dataset_path):
    print(f"Dataset Annotation Found: {dataset_path}")
else:
    print(f"Dataset Annotation Missing: {dataset_path}")

# =========================================================
# TEST IMAGE VALIDATION
# =========================================================

print("\n[12] Inference Images")

image_dir = "inference/images"

if os.path.exists(image_dir):

    images = [
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(images)} test images")

    for image in images:
        print(f" - {image}")

else:
    print(f"Missing Directory: {image_dir}")

print("\n" + "=" * 60)
print("Environment Verification Complete")
print("=" * 60)