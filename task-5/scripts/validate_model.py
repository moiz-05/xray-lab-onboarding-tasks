import onnx
import onnxruntime as ort
import numpy as np

ORIGINAL_MODEL = "../models/original/model_fp32.onnx"
SIMPLIFIED_MODEL = "../models/fixed/model_simplified.onnx"

INPUT_SHAPE = (1, 3, 224, 224)

np.random.seed(42)

dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

print("Running original model...")
sess_original = ort.InferenceSession(ORIGINAL_MODEL)

input_name_original = sess_original.get_inputs()[0].name

outputs_original = sess_original.run(
    None,
    {input_name_original: dummy_input}
)

print("Running simplified model...")
sess_simplified = ort.InferenceSession(SIMPLIFIED_MODEL)

input_name_simplified = sess_simplified.get_inputs()[0].name

outputs_simplified = sess_simplified.run(
    None,
    {input_name_simplified: dummy_input}
)

print("\nComparing outputs...")

for i, (orig, simp) in enumerate(zip(outputs_original, outputs_simplified)):
    mae = np.mean(np.abs(orig - simp))
    max_diff = np.max(np.abs(orig - simp))

    print(f"\nOutput {i}")
    print(f"MAE: {mae:.8f}")
    print(f"Max Difference: {max_diff:.8f}")

print("\nValidation complete.")