import onnx

INPUT_MODEL = "../models/original/model_fp32.onnx"
OUTPUT_MODEL = "../models/fixed/model_static.onnx"

STATIC_BATCH = 1
STATIC_HEIGHT = 224
STATIC_WIDTH = 224

model = onnx.load(INPUT_MODEL)

graph = model.graph

for input_tensor in graph.input:
    shape = input_tensor.type.tensor_type.shape

    shape.dim[0].dim_value = STATIC_BATCH
    shape.dim[2].dim_value = STATIC_HEIGHT
    shape.dim[3].dim_value = STATIC_WIDTH

print("Static input shape applied:")
print(f"[{STATIC_BATCH}, 3, {STATIC_HEIGHT}, {STATIC_WIDTH}]")

onnx.save(model, OUTPUT_MODEL)

print(f"Saved fixed model to: {OUTPUT_MODEL}")