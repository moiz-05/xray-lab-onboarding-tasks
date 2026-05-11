import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

# =========================================================
# CONFIGURATION
# =========================================================

ONNX_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "onnx",
    "models",
    "rtmdet_fruits.onnx"
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
    "onnx"
)

INPUT_SIZE = 640

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD ONNX MODEL
# =========================================================

print("=" * 60)
print("Loading ONNX Model")
print("=" * 60)

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime Session Initialized")

# =========================================================
# INPUT / OUTPUT NAMES
# =========================================================

input_name = session.get_inputs()[0].name

output_names = [
    output.name
    for output in session.get_outputs()
]

print(f"\nInput Name : {input_name}")

print(f"Output Names : {output_names}")

# =========================================================
# IMAGE FILES
# =========================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

print(f"\nFound {len(image_files)} images")

# =========================================================
# PROCESS IMAGES
# =========================================================

for image_name in image_files:

    print("\n" + "-" * 60)
    print(f"Processing: {image_name}")
    print("-" * 60)

    image_path = os.path.join(IMAGE_DIR, image_name)

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    original_image = cv2.imread(image_path)

    image = original_image.copy()

    original_h, original_w = image.shape[:2]

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    resized = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE)
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    normalized = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        normalized,
        (2, 0, 1)
    )

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    # -----------------------------------------------------
    # RUN INFERENCE
    # -----------------------------------------------------

    outputs = session.run(
        output_names,
        {input_name: tensor}
    )

    print("Inference Complete")

    # -----------------------------------------------------
    # DEBUG OUTPUTS
    # -----------------------------------------------------

    print("\nOutput Shapes:")

    for i, output in enumerate(outputs):

        print(f"Output {i}: {output.shape}")

    # =====================================================
    # VISUALIZE FEATURE ACTIVATIONS
    # =====================================================

    visualization = original_image.copy()

    # Use highest resolution feature map
    feature_map = outputs[0][0]

    # -----------------------------------------------------
    # CREATE HEATMAP
    # -----------------------------------------------------

    activation_map = np.max(
        feature_map,
        axis=0
    )

    activation_map = cv2.resize(
        activation_map,
        (original_w, original_h)
    )

    activation_map = cv2.normalize(
        activation_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    activation_map = activation_map.astype(np.uint8)

    heatmap = cv2.applyColorMap(
        activation_map,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        visualization,
        0.6,
        heatmap,
        0.4,
        0
    )

    # -----------------------------------------------------
    # FIND BEST REGION - Find strongest activation
    # -----------------------------------------------------

    activation_resized = cv2.resize(
        activation_map,
        (original_w, original_h)
    )

    # Find the point with maximum activation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(activation_resized)

    print(f"Max activation: {max_val} at {max_loc}")

    # Create binary mask for top activations - use higher threshold
    threshold = max_val * 0.8
    _, binary = cv2.threshold(
        activation_resized,
        int(threshold),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by area and select the best one
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            valid_contours.append((area, contour))

    # Sort by area descending and take the largest
    valid_contours.sort(key=lambda x: x[0], reverse=True)

    if valid_contours:
        best_contour = valid_contours[0][1]
        x, y, w, h = cv2.boundingRect(best_contour)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw circle at max activation point
        cv2.circle(overlay, max_loc, 10, (255, 0, 0), -1)

        label_text = f"Region ({w}x{h})"
        cv2.putText(
            overlay,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # -----------------------------------------------------
    # SAVE OUTPUT
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"onnx_{image_name}"
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    print(f"Saved Output: {output_path}")

# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("ONNX Runtime Inference Complete")
print("=" * 60)