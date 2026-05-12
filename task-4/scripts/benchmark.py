import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

from utils import setup_logger, save_json

logger = setup_logger()

# ---------------------------------------
# Configuration
# ---------------------------------------

MODEL_PATH = "../models/model.onnx"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
BATCH_SIZE = 1

DEFAULT_HEIGHT = 640
DEFAULT_WIDTH = 640

CSV_OUTPUT = "../benchmarks/benchmark_results.csv"
JSON_OUTPUT = "../benchmarks/cpu_benchmark.json"

# ---------------------------------------
# Load Model
# ---------------------------------------

logger.info("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_meta = session.get_inputs()[0]
input_name = input_meta.name
input_shape = input_meta.shape

logger.info(f"Input shape: {input_shape}")

# ---------------------------------------
# Resolve Dynamic Shape
# ---------------------------------------

shape = []

for idx, dim in enumerate(input_shape):

    if isinstance(dim, str) or dim is None:

        if idx == 0:
            shape.append(BATCH_SIZE)

        elif idx == 2:
            shape.append(DEFAULT_HEIGHT)

        elif idx == 3:
            shape.append(DEFAULT_WIDTH)

        else:
            shape.append(1)

    else:
        shape.append(dim)

logger.info(f"Resolved shape: {shape}")

# ---------------------------------------
# Dummy Input
# ---------------------------------------

dummy_input = np.random.randn(*shape).astype(np.float32)

# ---------------------------------------
# Warm-up
# ---------------------------------------

logger.info("Starting warm-up...")

for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: dummy_input})

logger.info("Warm-up complete.")

# ---------------------------------------
# Benchmark
# ---------------------------------------

logger.info("Running benchmark...")

latencies = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    outputs = session.run(
        None,
        {input_name: dummy_input}
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

logger.info("Benchmark complete.")

# ---------------------------------------
# Metrics
# ---------------------------------------

avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

p50_latency = np.percentile(latencies, 50)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)

fps = 1000 / avg_latency

results = {
    "runtime": "ONNXRuntime-CPU",
    "batch_size": BATCH_SIZE,
    "warmup_runs": WARMUP_RUNS,
    "benchmark_runs": BENCHMARK_RUNS,
    "average_latency_ms": float(avg_latency),
    "min_latency_ms": float(min_latency),
    "max_latency_ms": float(max_latency),
    "p50_latency_ms": float(p50_latency),
    "p95_latency_ms": float(p95_latency),
    "p99_latency_ms": float(p99_latency),
    "fps": float(fps)
}

# ---------------------------------------
# Print Results
# ---------------------------------------

logger.info("===== Benchmark Results =====")

for key, value in results.items():
    logger.info(f"{key}: {value}")

# ---------------------------------------
# CSV Export
# ---------------------------------------

df = pd.DataFrame([results])

csv_path = Path(CSV_OUTPUT)
csv_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(csv_path, index=False)

logger.info(f"CSV results saved to: {csv_path}")

# ---------------------------------------
# JSON Export
# ---------------------------------------

save_json(results, JSON_OUTPUT)

logger.info(f"JSON results saved to: {JSON_OUTPUT}")