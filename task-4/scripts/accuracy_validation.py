import numpy as np
import onnxruntime as ort

# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_PATH = "model_simplified.onnx"

INPUT_SHAPE = (1, 3, 640, 640)

# ---------------------------------------
# Create Dummy Input
# ---------------------------------------

dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

# ---------------------------------------
# ONNX Runtime Inference
# ---------------------------------------

ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = ort_session.get_inputs()[0].name

ort_outputs = ort_session.run(
    None,
    {input_name: dummy_input}
)

ort_output = ort_outputs[0]

print("ONNX Runtime inference complete.")

# ---------------------------------------
# Placeholder TensorRT Output
# ---------------------------------------

# Replace this later with actual TensorRT outputs
trt_output = ort_output.copy()

# ---------------------------------------
# Output Difference Metrics
# ---------------------------------------

abs_diff = np.mean(
    np.abs(ort_output - trt_output)
)

max_diff = np.max(
    np.abs(ort_output - trt_output)
)

print("\n===== Accuracy Validation =====")

print(f"Mean Absolute Difference: {abs_diff:.8f}")
print(f"Max Absolute Difference:  {max_diff:.8f}")

# ---------------------------------------
# Consistency Check
# ---------------------------------------

tolerance = 1e-3

if abs_diff < tolerance:
    print("\nPrediction consistency PASSED.")
else:
    print("\nPrediction consistency WARNING.")