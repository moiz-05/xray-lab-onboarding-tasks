import time
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = Path("../models/model_simplified.onnx")
RESULTS_PATH = Path("../benchmarks/cpu_baseline.json")

# ---------------------------------------
# Load ONNX Runtime Session
# ---------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

# ---------------------------------------
# Input Metadata
# ---------------------------------------

input_meta = session.get_inputs()[0]
input_name = input_meta.name
input_shape = input_meta.shape

print(f"Input Name: {input_name}")
print(f"Input Shape: {input_shape}")

shape = []

for idx, dim in enumerate(input_shape):

    if isinstance(dim, str) or dim is None:

        # Batch dimension
        if idx == 0:
            shape.append(1)

        # Height / Width dimensions
        else:
            shape.append(640)

    else:
        shape.append(dim)

# ---------------------------------------
# Generate Dummy Input
# ---------------------------------------

dummy_input = np.random.randn(*shape).astype(np.float32)

# ---------------------------------------
# Warm-up
# ---------------------------------------

print("\nRunning warm-up...")

for _ in range(10):
    session.run(None, {input_name: dummy_input})

print("Warm-up complete.")

# ---------------------------------------
# Benchmark
# ---------------------------------------

runs = 100
latencies = []

print("\nRunning benchmark...")

for _ in range(runs):
    start = time.perf_counter()

    outputs = session.run(
        None,
        {input_name: dummy_input}
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

# ---------------------------------------
# Metrics
# ---------------------------------------

avg_latency = np.mean(latencies)
p50_latency = np.percentile(latencies, 50)
p95_latency = np.percentile(latencies, 95)
fps = 1000 / avg_latency

print("\n===== Benchmark Results =====")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"P50 Latency:     {p50_latency:.2f} ms")
print(f"P95 Latency:     {p95_latency:.2f} ms")
print(f"FPS:             {fps:.2f}")

# ---------------------------------------
# Output Validation
# ---------------------------------------

print("\n===== Output Validation =====")

for idx, output in enumerate(outputs):
    print(f"Output {idx}: shape={output.shape}")

print("\nModel inference successful.")

# ---------------------------------------
# Save Results
# ---------------------------------------

results = {
    "average_latency_ms": float(avg_latency),
    "p50_latency_ms": float(p50_latency),
    "p95_latency_ms": float(p95_latency),
    "fps": float(fps),
    "runs": runs
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {RESULTS_PATH}")