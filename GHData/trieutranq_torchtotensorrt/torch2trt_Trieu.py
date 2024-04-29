# asta | Jan 31 21

import time
import torch
import cv2
from pathlib import Path
# from torch2trt import TRTModule
# from torch2trt import torch2trt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import random
from typing import Callable
from torch import nn


def help_preproduce(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


help_preproduce(51120)


from torchvision import models

ARCHS = {"resnet18": models.resnet18, "resnet50": models.resnet50}


class BaseModel(nn.Module):
    def __init__(
        self,
        arch_: Callable,
        num_classes_: int,
        nb_channels_: int,
        pretrained_: bool,
    ) -> nn.Module:
        super().__init__()
        self.base_model = models.resnet50(pretrained=pretrained_)
        self.base_model.conv1 = nn.Conv2d(
            nb_channels_,
            64,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(0, 0),
            bias=False,
        )
        self.appended = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_),
        )

    def forward(self, input_image_):
        output_model = self.base_model(input_image_)
        return self.appended(output_model)


base_model = BaseModel(
    arch_="resnet50",
    num_classes_=37,
    nb_channels_=3,
    pretrained_=False,
)

# model_dir_ = (
#     old_model_name
# ) = "/media/Data/source_Mark_ML_asta/ADI/DLEngine/models/asta_test_LeeTop/source_NUC_markML/sample/models/terry_ADLT_chars/char_adlt.pt"
model_dir_ = (
    old_model_name
) = "/home/timtran/Desktop/TensorRT/char_adlt.pt"

base_model.load_state_dict(torch.load(model_dir_, map_location="cpu"))


# image_shape input = 64x64
# grayscale, nb channels = 3
# transfroms.GrayScale(outchannel=3)


base_model.eval().cuda()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# base_model.to(device)

# Evaluate
# import glob
# from torch.utils.data import DataLoader
# from torchvision import transforms
# def load_data(path, shape = (64,64)):
#     data = list()
#     label = list()
#     transform = transforms.Compose([
#         transforms.Resize((64,64)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor()]
#     )
#     for i in path:
#         image = Image.open(i)
#         image = transform(image)
#         data.append(image)
#         label.append(1)
#     return data, label

# path =  sorted(glob.glob('/home/timtran/Desktop/TensorRT/20220331_ADLT_MergeLegacy_Train/1/*'))
# x_data, x_label = load_data(path, shape = (64,64))
# train_data = []
# for i in range(len(x_data)):
#    train_data.append([x_data[i], x_label[i]])
# # print(train_data)
# data = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0)

# def evaluate(data):
#   correct =0
#   base_model.eval()
#   for i, (inputs, labels) in enumerate(data, 0):
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     # result = session.run([output_name], {input_name: inputs})
#     result = base_model(inputs)
#     # result = np.array(result).squeeze()
#     _,preds = torch.max(result.squeeze(), 1)
#     print(preds)
#     correct += torch.sum(preds == labels)
#     print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(data.dataset),100. * correct / len(data.dataset)))

# evaluate(data)

ONNX_FILE_PATH = 'char_adlt.onnx'

def convert_onnx(model):
    inputs = ['images']
    outputs = ['result']
    dynamic_axes =  {'images': {0: 'batch'}, 'result': {0: 'batch'}}
    torch.onnx.export(model, torch.randn(1, 3, 64, 64).cuda(), ONNX_FILE_PATH, input_names= inputs,
                    output_names= outputs, export_params=True,verbose=True, opset_version = 13, dynamic_axes=dynamic_axes)
    print('Converted to ONNX')

# convert_onnx(base_model)
import tensorrt as trt

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
        profile.set_shape("images", (1,3,64,64), (32,3,64,64), (128,3,64,64)) 
        config.add_optimization_profile(profile)
        builder.max_batch_size = 128

        # config.set_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.INT8)
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

convert_tensorrt(onnx_file_path='/home/timtran/Desktop/TensorRT/char_adlt.onnx',engine_file_path= '/home/timtran/Desktop/TensorRT/char_adlt-int8.engine')