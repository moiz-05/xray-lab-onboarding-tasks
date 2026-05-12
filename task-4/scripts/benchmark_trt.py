import time
import numpy as np
import pandas as pd
from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ---------------------------------------
# Configuration
# ---------------------------------------

ENGINE_PATH = "model_fp16.engine"

WARMUP_RUNS = 50
BENCHMARK_RUNS = 200

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch",
    type=int,
    default=1
)

args = parser.parse_args()

BATCH_SIZE = args.batch

INPUT_SHAPE = (BATCH_SIZE, 3, 640, 640)

CSV_OUTPUT = "trt_benchmark_results.csv"

# ---------------------------------------
# TensorRT Runtime
# ---------------------------------------

LOGGER = trt.Logger(trt.Logger.WARNING)

runtime = trt.Runtime(LOGGER)

# ---------------------------------------
# Load Engine
# ---------------------------------------

with open(ENGINE_PATH, "rb") as f:
    engine_data = f.read()

engine = runtime.deserialize_cuda_engine(engine_data)

context = engine.create_execution_context()

print("TensorRT engine loaded successfully.")

# ---------------------------------------
# Allocate Buffers
# ---------------------------------------

input_size = np.prod(INPUT_SHAPE)

dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

# GPU memory allocation
d_input = cuda.mem_alloc(dummy_input.nbytes)

# Output tensor handling
output_shape = (1, 25200, 85)  # Example YOLO output
output = np.empty(output_shape, dtype=np.float32)

d_output = cuda.mem_alloc(output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()




input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

context.set_tensor_address(
    input_name,
    int(d_input)
)

context.set_tensor_address(
    output_name,
    int(d_output)
)

# ---------------------------------------
# Warm-Up
# ---------------------------------------

print("\nRunning warm-up...")

warmup_latencies = []

for _ in range(WARMUP_RUNS):

    start = time.perf_counter()

    cuda.memcpy_htod_async(
        d_input,
        dummy_input,
        stream
    )

    context.set_tensor_address(
        engine.get_tensor_name(0),
        int(d_input)
    )

    context.set_tensor_address(
        engine.get_tensor_name(1),
        int(d_output)
    )

    context.execute_async_v3(
        stream_handle=stream.handle
    )

    cuda.memcpy_dtoh_async(
        output,
        d_output,
        stream
    )

    stream.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    warmup_latencies.append(latency_ms)

print("Warm-up complete.")

# ---------------------------------------
# Warm-Up Analysis
# ---------------------------------------

print("\nWarm-up latency trend:")

for idx, latency in enumerate(warmup_latencies[:10]):
    print(f"Run {idx+1}: {latency:.2f} ms")

# ---------------------------------------
# Benchmark
# ---------------------------------------

print("\nRunning steady-state benchmark...")

latencies = []

gpu_memory_before = cuda.mem_get_info()

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    cuda.memcpy_htod_async(
        d_input,
        dummy_input,
        stream
    )

    context.execute_async_v3(
        stream_handle=stream.handle
    )

    cuda.memcpy_dtoh_async(
        output,
        d_output,
        stream
    )

    stream.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

gpu_memory_after = cuda.mem_get_info()

print("Benchmark complete.")

# ---------------------------------------
# Metrics
# ---------------------------------------

avg_latency = np.mean(latencies)

p50_latency = np.percentile(latencies, 50)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)

fps = 1000 / avg_latency

used_memory_mb = (
    (gpu_memory_before[0] - gpu_memory_after[0])
    / (1024 * 1024)
)

results = {
    "engine": ENGINE_PATH,
    "average_latency_ms": float(avg_latency),
    "p50_latency_ms": float(p50_latency),
    "p95_latency_ms": float(p95_latency),
    "p99_latency_ms": float(p99_latency),
    "fps": float(fps),
    "gpu_memory_used_mb": float(used_memory_mb)
}

# ---------------------------------------
# Print Results
# ---------------------------------------

print("\n===== TensorRT Benchmark Results =====")

for key, value in results.items():
    print(f"{key}: {value}")

# ---------------------------------------
# Save Results
# ---------------------------------------

df = pd.DataFrame([results])

df.to_csv(CSV_OUTPUT, index=False)

print(f"\nResults saved to: {CSV_OUTPUT}")