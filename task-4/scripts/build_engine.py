from pathlib import Path

import tensorrt as trt

# ---------------------------------------
# Configuration
# ---------------------------------------

ONNX_PATH = "model_simplified.onnx"

FP32_ENGINE_PATH = "model_fp32.engine"
FP16_ENGINE_PATH = "model_fp16.engine"

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--workspace",
    type=int,
    default=4,
    help="Workspace size in GB"
)

args = parser.parse_args()

WORKSPACE_SIZE = args.workspace << 30

LOGGER = trt.Logger(trt.Logger.VERBOSE)

# ---------------------------------------
# Engine Builder
# ---------------------------------------

def build_engine(
    onnx_path,
    engine_path,
    use_fp16=False
):

    builder = trt.Builder(LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, LOGGER)

    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        WORKSPACE_SIZE
    )

    # ---------------------------------------
    # FP16 Support
    # ---------------------------------------

    if use_fp16:

        if builder.platform_has_fast_fp16:
            print("\nFP16 supported on this GPU.")
            config.set_flag(trt.BuilderFlag.FP16)

        else:
            print("\nFP16 NOT supported.")

    # ---------------------------------------
    # Parse ONNX
    # ---------------------------------------

    print(f"\nLoading ONNX model: {onnx_path}")

    with open(onnx_path, "rb") as model:

        if not parser.parse(model.read()):

            print("\nERROR: Failed to parse ONNX.")

            for idx in range(parser.num_errors):
                print(parser.get_error(idx))

            return None

    # ---------------------------------------
    # Dynamic Shape Profile
    # ---------------------------------------

    profile = builder.create_optimization_profile()

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    print(f"\nInput Tensor Name: {input_name}")

    profile.set_shape(
        input_name,
        min=(1, 3, 320, 320),
        opt=(1, 3, 640, 640),
        max=(8, 3, 1280, 1280)
    )

    config.add_optimization_profile(profile)

    # ---------------------------------------
    # Build Engine
    # ---------------------------------------

    print("\nBuilding TensorRT engine...")

    serialized_engine = builder.build_serialized_network(
        network,
        config
    )

    if serialized_engine is None:
        print("\nEngine build failed.")
        return None

    # ---------------------------------------
    # Save Engine
    # ---------------------------------------

    engine_path = Path(engine_path)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"\nEngine saved to: {engine_path}")

    # ---------------------------------------
    # Engine Size
    # ---------------------------------------

    size_mb = engine_path.stat().st_size / (1024 * 1024)

    print(f"Engine size: {size_mb:.2f} MB")

    return engine_path

# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    print("\n===== FP32 ENGINE BUILD =====")

    build_engine(
        ONNX_PATH,
        FP32_ENGINE_PATH,
        use_fp16=False
    )

    print("\n===== FP16 ENGINE BUILD =====")

    build_engine(
        ONNX_PATH,
        FP16_ENGINE_PATH,
        use_fp16=True
    )