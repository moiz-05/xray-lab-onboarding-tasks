import onnx
from onnx import shape_inference
from collections import Counter
import os

# MODEL_PATH = "../models/original/model_fp32.onnx"
# MODEL_PATH = "../models/fixed/model_static.onnx"
MODEL_PATH = "../models/fixed/model_simplified.onnx"

SUPPORTED_OPS = {
    "Conv",
    "Relu",
    "Add",
    "Mul",
    "BatchNormalization",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
    "MatMul",
    "Concat",
    "Transpose",
    "Reshape",
    "Sigmoid",
    "Softmax"
}

def get_shape(tensor):
    dims = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(dim.dim_value)
    return dims

def main():
    print("=" * 60)
    print("Loading ONNX model...")
    print("=" * 60)

    model = onnx.load(MODEL_PATH)

    print("\nRunning shape inference...")
    inferred_model = shape_inference.infer_shapes(model)

    graph = inferred_model.graph

    print("\n" + "=" * 60)
    print("MODEL INFO")
    print("=" * 60)

    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model Size: {model_size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("INPUT TENSORS")
    print("=" * 60)

    for inp in graph.input:
        print(f"Name: {inp.name}")
        print(f"Shape: {get_shape(inp)}")
        print(f"Type: {inp.type.tensor_type.elem_type}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("OUTPUT TENSORS")
    print("=" * 60)

    for out in graph.output:
        print(f"Name: {out.name}")
        print(f"Shape: {get_shape(out)}")
        print(f"Type: {out.type.tensor_type.elem_type}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("GRAPH ANALYSIS")
    print("=" * 60)

    op_counts = Counter()

    unsupported_ops = set()

    for node in graph.node:
        op_counts[node.op_type] += 1

        if node.op_type not in SUPPORTED_OPS:
            unsupported_ops.add(node.op_type)

    total_nodes = sum(op_counts.values())

    print(f"Total Nodes: {total_nodes}")

    print("\nOperator Counts:")
    for op, count in sorted(op_counts.items()):
        print(f"{op}: {count}")

    print("\n" + "=" * 60)
    print("DYNAMIC SHAPE ANALYSIS")
    print("=" * 60)

    dynamic_found = False

    for inp in graph.input:
        shape = get_shape(inp)

        for dim in shape:
            if isinstance(dim, str):
                dynamic_found = True
                print(f"Dynamic dimension found in input '{inp.name}': {dim}")

    if not dynamic_found:
        print("No dynamic dimensions detected.")

    print("\n" + "=" * 60)
    print("UNSUPPORTED OPERATORS")
    print("=" * 60)

    if unsupported_ops:
        for op in sorted(unsupported_ops):
            print(op)
    else:
        print("No unsupported ops detected.")

    print("\nInspection complete.")

if __name__ == "__main__":
    main()