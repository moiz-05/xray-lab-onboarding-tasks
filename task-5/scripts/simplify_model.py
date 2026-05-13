import onnx
from onnxsim import simplify

INPUT_MODEL = "../models/fixed/model_static.onnx"
OUTPUT_MODEL = "../models/fixed/model_simplified.onnx"

print("Loading model...")
model = onnx.load(INPUT_MODEL)

print("Simplifying model...")
model_simplified, check = simplify(model)

if check:
    print("Simplification successful.")
else:
    print("Simplification failed.")

onnx.save(model_simplified, OUTPUT_MODEL)

print(f"Saved simplified model to: {OUTPUT_MODEL}")