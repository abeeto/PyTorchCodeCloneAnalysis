from json.tool import main
import torch
from torchvision.models import resnet18
import torch.nn as nn
import tensorrt as trt
import os

ONNX_FILE_PATH = 'resnet18-mnist.onnx'


model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.load_state_dict(torch.load('/home/timtran/Desktop/TensorRT/resnet18-mnist.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def convert_onnx(model):
    inputs = ['images']
    outputs = ['result']
    dynamic_axes =  {'images': {0: 'batch'}, 'result': {0: 'batch'}}
    torch.onnx.export(model, torch.randn(1, 1, 28, 28).cuda(), ONNX_FILE_PATH, input_names= inputs,
                    output_names= outputs, export_params=True, opset_version = 12, dynamic_axes=dynamic_axes)
    print('Converted to ONNX')


def convert_tensorrt(onnx_file_path, engine_file_path):       
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        builder = trt.Builder(TRT_LOGGER)
        
        # network = builder.create_network(common.EXPLICIT_BATCH)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # input = network.add_input("images", trt.DataType.FLOAT, (-1,28,28))
        # identity = network.add_identity(input)
        # output = identity.get_output(0)
        # output.name = "result"
        # network.mark_output(output)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        # runtime = trt.Runtime(TRT_LOGGER)

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')
        
        # Print input info
        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)
        # Dynamic Shape
        config = builder.create_builder_config()

        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1,1,28,28), (32,1,28,28), (128,1,28,28)) 
        config.add_optimization_profile(profile)
        builder.max_batch_size = 128

        config.set_flag(trt.BuilderFlag.TF32)
        config.max_workspace_size = 4 << 20 # 256MiB

        print('Building an engine from file {}; this may take a while...'.format(
            onnx_file_path))
        # plan = builder.build_serialized_network(network, config)
        # print(plan)
        # engine = runtime.deserialize_cuda_engine(plan)
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        print(engine)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

    # if os.path.exists(engine_file_path):
    #     # If a serialized engine exists, use it instead of building an engine.
    #     print("Reading engine from file {}".format(engine_file_path))
    #     with open(engine_file_path, "rb") as f:
    #         runtime = trt.Runtime(TRT_LOGGER)
    #         return runtime.deserialize_cuda_engine(f.read())
    # else:
    build_engine()

convert_tensorrt(onnx_file_path='/home/timtran/Desktop/TensorRT/resnet18-mnist.onnx',engine_file_path= '/home/timtran/Desktop/TensorRT/resnet18-mnist.engine')
