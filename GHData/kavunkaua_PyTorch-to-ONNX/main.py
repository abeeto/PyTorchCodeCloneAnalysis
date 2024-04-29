import onnx
import torch
import torchvision
from onnx2pytorch import ConvertModel

#onnx_model = onnx.load("models/TrophNet.onnx")
onnx_model = onnx.load("models/squeezenet1.1-7.onnx")
pytorch_model = ConvertModel(onnx_model)

#dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
#input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
#output_names = [ "output1" ]
#torch.onnx.export(pytorch_model, dummy_input, "/home/serhii/VSCProjects/trophallaxis-DATA/TrophNet2.onnx", export_params=True, opset_version=8, do_constant_folding=True, input_names=input_names, output_names=output_names)