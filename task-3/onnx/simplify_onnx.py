import os
import onnx
from onnxsim import simplify

# =========================================================
# PATHS
# =========================================================

INPUT_MODEL = "models/rtmdet_fruits.onnx"

OUTPUT_MODEL = "models/rtmdet_fruits_simplified.onnx"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Loading ONNX Model")
print("=" * 60)

model = onnx.load(INPUT_MODEL)

print("Model Loaded")

# =========================================================
# SIMPLIFY
# =========================================================

print("\nSimplifying ONNX Model...")

simplified_model, check = simplify(model)

# =========================================================
# VALIDATE
# =========================================================

if check:

    onnx.save(
        simplified_model,
        OUTPUT_MODEL
    )

    print("Simplification Successful")

    print(f"Saved Simplified Model: {OUTPUT_MODEL}")

else:

    print("Simplification Failed")

print("\n" + "=" * 60)
print("ONNX Simplification Complete")
print("=" * 60)