import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
from time import time
from torch2onnx import ONNX_FILE_PATH, preprocess_image, postprocess

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder_config = builder.create_builder_config()
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder_config.max_workspace_size = 1 << 30
    # use FP16 mode if possible
    builder_config.set_flag(trt.BuilderFlag.FP16)

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_engine(network, builder_config)
    if engine is None:
        raise "Cannot create engine"
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context


# initialize TensorRT engine and parse ONNX model
engine, context = build_engine(ONNX_FILE_PATH)

# get sizes of input and output and allocate memory required for input data and for output data
for binding in engine:
    if engine.binding_is_input(binding):  # we expect only one input
        input_shape = engine.get_binding_shape(binding)
        input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
        device_input = cuda.mem_alloc(input_size)
    else:  # and one output
        output_shape = engine.get_binding_shape(binding)
        # create page-locked memory buffers (i.e. won't be swapped to disk)
        host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
        device_output = cuda.mem_alloc(host_output.nbytes)

# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()
# preprocess input data
host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
cuda.memcpy_htod_async(device_input, host_input, stream)
# run inference
start = time()
context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
end = time()
print("inf time: ", end - start)
cuda.memcpy_dtoh_async(host_output, device_output, stream)
stream.synchronize()
# postprocess results
output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
postprocess(output_data)
