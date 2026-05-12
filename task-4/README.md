# ONNX to TensorRT — GPU Optimization

Convert ONNX models into optimized TensorRT engines and benchmark GPU inference performance across FP32, FP16, and INT8 precision modes. This project follows a **CPU + Cloud GPU** workflow: local machine handles ONNX validation, benchmarking framework, and script preparation; a cloud GPU environment (Google Colab / Kaggle / RunPod) handles TensorRT conversion and GPU benchmarking.

---

## Objective

- Validate ONNX model inference on CPU (ONNX Runtime)
- Convert ONNX → TensorRT engines (FP32, FP16, optional INT8)
- Benchmark latency, throughput, and GPU memory across precision modes
- Tune optimization parameters: workspace memory, batch size, dynamic shapes
- Compare accuracy between ONNX CPU and TensorRT GPU outputs
- Identify the optimal deployment configuration

---

## Directory Structure

```
day4_tensorrt_optimization/
├── models/              # ONNX model files (.onnx)
├── engines/             # Serialized TensorRT engines (.engine)
├── scripts/             # Conversion, benchmark, validation, visualization scripts
├── benchmarks/          # CPU/GPU benchmark output files (JSON, CSV)
├── results/             # Graphs, charts, latency tables
├── configs/             # Benchmark configuration (JSON)
├── logs/                # Build and benchmark logs
├── notebooks/           # Google Colab / Jupyter notebooks
├── requirements.txt     # Python dependencies
├── .gitignore           # Gitignore rules
└── README.md            # This file
```

---

## Setup

### Local Environment (CPU)

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Key packages:
- `onnx`, `onnxruntime` — ONNX model loading and CPU inference
- `numpy`, `pandas` — Data handling and metrics
- `matplotlib` — Result visualization
- `opencv-python` — Image preprocessing
- `tensorrt`, `pycuda` — TensorRT conversion and GPU inference (cloud GPU only)

### Cloud GPU Environment

See the [notebooks](./notebooks) directory for Colab-compatible notebooks. The cloud environment requires:
- NVIDIA GPU + CUDA + TensorRT installed
- `trtexec` CLI tool available
- Upload `models/`, `scripts/`, and `configs/` to the cloud instance

---

## Workflow

### Phase 1 — ONNX CPU Validation

Validate the ONNX model with ONNX Runtime on CPU:

```bash
cd scripts
python validate_onnx_cpu.py
```

This loads the model, performs a warm-up loop, runs 100 benchmark iterations, and saves results to `benchmarks/cpu_baseline.json`.

### Phase 2 — CPU Benchmarking Framework

A reusable benchmarking script with configurable warm-up iterations, benchmark iterations, and batch size:

```bash
cd scripts
python benchmark.py
```

Outputs:
- `benchmarks/cpu_benchmark.json` — Latency metrics (avg, p50, p95, p99, FPS)
- `benchmarks/benchmark_results.csv` — Tabular results

### Phase 3 — Engine Building (on cloud GPU)

Build TensorRT engines from the ONNX model. Supports FP32, FP16, and configurable workspace size.

```bash
cd scripts
python build_engine.py --workspace 4
```

Or using `trtexec` CLI:

```bash
# FP32
trtexec --onnx=models/model_simplified.onnx --saveEngine=engines/model_fp32.engine --workspace=4096 --verbose

# FP16
trtexec --onnx=models/model_simplified.onnx --saveEngine=engines/model_fp16.engine --fp16 --workspace=4096 --verbose

# Dynamic shapes
trtexec --onnx=models/model_simplified.onnx --saveEngine=engines/model_dynamic_fp16.engine --fp16 --minShapes=input:1x3x320x320 --optShapes=input:1x3x640x640 --maxShapes=input:8x3x1280x1280 --workspace=4096 --verbose
```

Refer to [scripts/trtexec_commands.md](scripts/trtexec_commands.md) for the full list of conversion commands.

### Phase 4 — GPU Benchmarking

Benchmark TensorRT engines with CUDA synchronization timing:

```bash
cd scripts
python benchmark_trt.py --batch 1
```

Benchmarks run with configurable batch sizes (1, 2, 4, 8), warm-up iterations (50), and steady-state iterations (200). GPU memory usage is captured before and after each run.

### Phase 5 — Accuracy Validation

Compare ONNX CPU outputs against TensorRT GPU outputs:

```bash
cd scripts
python accuracy_validation.py
```

Reports mean absolute difference and max absolute difference between precision modes.

### Phase 6 — Visualization

Generate FPS comparison charts, latency percentile charts, and summary tables:

```bash
cd scripts
python visualize_results.py
```

Charts and tables are saved to the `results/` directory.

---

## Scripts Reference

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `validate_onnx_cpu.py` | Load ONNX model, warm-up, benchmark, validate outputs | onnxruntime |
| `benchmark.py` | Reusable CPU benchmark framework with CSV/JSON export | onnxruntime, pandas |
| `build_engine.py` | Convert ONNX → TensorRT engine (FP32/FP16) | tensorrt |
| `benchmark_trt.py` | Benchmark TensorRT engines with CUDA timing | tensorrt, pycuda |
| `accuracy_validation.py` | Compare ONNX vs TensorRT output accuracy | onnxruntime, numpy |
| `visualize_results.py` | Generate latency/FPS charts and tables | pandas, matplotlib |
| `utils.py` | Shared logging and JSON export utilities | — |
| `trtexec_commands.md` | Reference for trtexec CLI commands | — |

---

## Configuration

Edit [configs/benchmark_config.json](configs/benchmark_config.json):

```json
{
    "warmup_runs": 100,
    "benchmark_runs": 500,
    "batch_sizes": [1, 2, 4, 8],
    "default_height": 640,
    "default_width": 640,
    "workspace_size_gb": 4,
    "enable_fp16": true
}
```

---

## Benchmark Results

### CPU Baseline (ONNX Runtime)

| Metric | Value |
|--------|-------|
| Average Latency | 236.43 ms |
| P50 Latency | 225.56 ms |
| P95 Latency | 332.33 ms |
| FPS | 4.23 |

### TensorRT GPU Results

GPU benchmark results across precision modes and batch sizes are stored in `results/` as CSV files (`trt_fp32_b{N}_benchmark_results.csv`, `trt_fp16_b{N}_benchmark_results.csv`). Each file contains average latency, p50/p95/p99 latency, FPS, and GPU memory usage.

---

## Cloud GPU Quick Start

1. Upload the `models/`, `scripts/`, and `configs/` directories to your cloud instance
2. Install TensorRT and required Python packages
3. Build engines: `python scripts/build_engine.py --workspace 4`
4. Benchmark: `python scripts/benchmark_trt.py --batch 1`
5. Validate accuracy: `python scripts/accuracy_validation.py`
6. Download results back to the local `results/` directory

---

## Precision Modes

| Mode | Description |
|------|-------------|
| FP32 | Full precision, highest accuracy, baseline |
| FP16 | Half precision, ~2x throughput, minimal accuracy loss |
| INT8 | Integer quantization (optional), highest throughput, requires calibration |

---

## License

This project is part of the xray-labs GPU optimization curriculum.
