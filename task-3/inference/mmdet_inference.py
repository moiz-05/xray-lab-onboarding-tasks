import os
import sys

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

from mmdet.apis import init_detector
from mmdet.apis import inference_detector

from mmengine.registry import VISUALIZERS

from utils.preprocessing import load_image

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

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "inference",
    "images"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "inference",
    "outputs",
    "mmdet"
)

SCORE_THRESHOLD = 0.45

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Initializing MMDetection Model")
print("=" * 60)

model = init_detector(
    CONFIG_FILE,
    CHECKPOINT_FILE,
    device="cpu"
)

print("Model Loaded Successfully")

# =========================================================
# CREATE VISUALIZER
# =========================================================

visualizer = VISUALIZERS.build(model.cfg.visualizer)

visualizer.dataset_meta = model.dataset_meta

# =========================================================
# GET TEST IMAGES
# =========================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

print(f"\nFound {len(image_files)} test images")

# =========================================================
# RUN INFERENCE
# =========================================================

for image_name in image_files:

    print("\n" + "-" * 60)
    print(f"Processing: {image_name}")
    print("-" * 60)

    image_path = os.path.join(IMAGE_DIR, image_name)

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = load_image(image_path)

    # -----------------------------------------------------
    # RUN INFERENCE
    # -----------------------------------------------------

    result = inference_detector(model, image)

    print("Inference Complete")

    # -----------------------------------------------------
    # SAVE VISUALIZATION
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"mmdet_{image_name}"
    )

    visualizer.add_datasample(
        name="result",
        image=image,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=SCORE_THRESHOLD,
        out_file=output_path
    )

    print(f"Saved Output: {output_path}")

# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("MMDetection Inference Completed")
print("=" * 60)