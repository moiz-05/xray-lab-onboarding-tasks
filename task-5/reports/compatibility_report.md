# Hailo Compatibility Report

## Model Information

| Property | Value |
|---|---|
| Model Name | model_fp32.onnx |
| Model Size | 38.32 MB |
| Total Nodes | 301 |
| Input Name | input |
| Input Shape | ['batch_size', 3, 'height', 'width'] |
| Output Count | 6 |
| Framework Format | ONNX |

---

# Input / Output Analysis

## Input Tensor

```text
input: ['batch_size', 3, 'height', 'width']
```

### Observations

The model currently uses dynamic dimensions:

- batch_size
- height
- width

This is problematic for Hailo compilation because Hailo generally requires:

- static batch size
- static spatial dimensions

### Required Fix

The model must be converted to a static input shape.

Recommended example:

```text
[1, 3, 224, 224]
```

Actual dimensions depend on the original training/export configuration.

---

# Output Tensor Analysis

The model exposes 6 output tensors:

| Output Name | Shape |
|---|---|
| output | ['batch_size', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |
| 860 | ['Conv860_dim_0', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |
| 880 | ['Conv880_dim_0', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |
| 851 | ['Mul851_dim_0', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |
| 871 | ['Mul871_dim_0', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |
| 891 | ['Mul891_dim_0', 4, 'Convoutput_dim_2', 'Convoutput_dim_3'] |

### Observations

- Output dimensions are also dynamic.
- Several unnamed intermediate outputs exist.
- The graph likely exports internal tensors unintentionally.

### Potential Issue

Hailo compilation is usually more stable when:
- only final outputs are exported
- intermediate tensors are removed

---

# Operator Analysis

| Operator | Count | Hailo Compatibility |
|---|---|---|
| Conv | 92 | Supported |
| Sigmoid | 82 | Supported |
| Mul | 89 | Supported |
| Concat | 13 | Usually Supported |
| Add | 5 | Supported |
| GlobalAveragePool | 4 | Supported |
| MaxPool | 3 | Supported |
| Constant | 7 | Potentially Problematic |
| HardSigmoid | 4 | Potentially Problematic |
| Resize | 2 | Frequently Problematic |

---

# Unsupported / Risky Operators

## Resize

### Risk Level
HIGH

### Reason

Resize operations are commonly problematic on edge accelerators because:
- interpolation modes may not be supported
- dynamic scaling may fail
- hardware mapping may be inefficient

### Recommendation

Possible fixes:
- replace Resize with supported upsampling
- use nearest-neighbor resize
- simplify graph
- fold resize into preprocessing if possible

---

## HardSigmoid

### Risk Level
MEDIUM

### Reason

Some Hailo compiler versions have partial support for HardSigmoid.

### Recommendation

Potential replacements:
- standard Sigmoid
- ReLU6-based approximation
- graph simplification

Further testing required.

---

## Constant

### Risk Level
LOW

### Reason

Constants are usually harmless but may indicate:
- embedded shape logic
- dynamic graph behavior

Requires parser validation.

---

# Dynamic Shape Analysis

## Detected Dynamic Dimensions

| Tensor | Dynamic Dimensions |
|---|---|
| input | batch_size, height, width |

### Risk Level
CRITICAL

### Reason

Hailo compilation generally requires fully static input dimensions.

### Required Fix

Convert:

```text
['batch_size', 3, 'height', 'width']
```

to something like:

```text
[1, 3, 224, 224]
```

---

# Graph Complexity Assessment

| Metric | Value |
|---|---|
| Total Nodes | 301 |
| Conv Layers | 92 |
| Activation Layers | 86+ |
| Resize Layers | 2 |

### Assessment

The graph appears to be:

- CNN-based
- moderately large
- likely deployable after graph fixes

No transformer-style operators detected.

This is a positive sign for Hailo compatibility.

---

# Major Compatibility Risks

| Issue | Severity |
|---|---|
| Dynamic input dimensions | CRITICAL |
| Resize operators | HIGH |
| Multiple intermediate outputs | MEDIUM |
| HardSigmoid ops | MEDIUM |

---

# Recommended Next Steps

## Phase 3 — Model Surgery

Required actions:

- Fix dynamic input dimensions
- Export static ONNX graph
- Simplify ONNX graph
- Remove unnecessary outputs
- Validate Resize operator behavior
- Re-test ONNX correctness

---

# Expected Probability of Successful Hailo Compilation

| Stage | Estimated Success |
|---|---|
| Current Model | LOW |
| After Static Shape Fix | MEDIUM |
| After Graph Simplification | GOOD |
| After Resize Handling | HIGH |

---

# Final Assessment

The model is likely compatible with Hailo hardware after moderate graph surgery.

Primary blockers are:

1. Dynamic input dimensions
2. Resize operations
3. Multiple intermediate outputs

The overall operator set is mostly CNN-friendly and suitable for edge deployment.