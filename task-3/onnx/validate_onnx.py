import os
import onnx

# =========================================================
# PATH
# =========================================================

ONNX_MODEL_PATH = "models/rtmdet_fruits.onnx"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("Loading ONNX Model")
print("=" * 60)

model = onnx.load(ONNX_MODEL_PATH)

print("ONNX Model Loaded")

# =========================================================
# VALIDATE MODEL
# =========================================================

print("\nValidating ONNX Graph...")

onnx.checker.check_model(model)

print("ONNX Graph Validation Successful")

# =========================================================
# GRAPH INFO
# =========================================================

print("\nGraph Information")

print(f"IR Version : {model.ir_version}")

print(f"Producer   : {model.producer_name}")

print(f"Opsets     : {model.opset_import}")

print("\n" + "=" * 60)
print("ONNX Validation Complete")
print("=" * 60)