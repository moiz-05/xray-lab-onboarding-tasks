# TensorRT Conversion Commands

---

# FP32 Engine Build

```bash
trtexec \
--onnx=models/model_simplified.onnx \
--saveEngine=engines/model_fp32.engine \
--workspace=4096 \
--verbose
```

---

# FP16 Engine Build

```bash
trtexec \
--onnx=models/model_simplified.onnx \
--saveEngine=engines/model_fp16.engine \
--fp16 \
--workspace=4096 \
--verbose
```

---

# Dynamic Shape FP16 Build

```bash
trtexec \
--onnx=models/model_simplified.onnx \
--saveEngine=engines/model_dynamic_fp16.engine \
--fp16 \
--minShapes=input:1x3x320x320 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:8x3x1280x1280 \
--workspace=4096 \
--verbose
```

---

# Benchmark Existing Engine

```bash
trtexec \
--loadEngine=engines/model_fp16.engine \
--shapes=input:1x3x640x640 \
--warmUp=100 \
--iterations=500
```