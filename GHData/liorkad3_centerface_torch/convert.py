import torch
import torchsummary
from src.centerface import Centerface

def convert_to_onnx(model, onnx_name):
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    torch_out = model(x)
    torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                onnx_name,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['heatmap', 'scale', 'offset', 'landmarks'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'heatmap' : {0 : 'batch_size'},
                                'scale' : {0 : 'batch_size'},
                                'offset' : {0 : 'batch_size'},
                                'landmarks' : {0 : 'batch_size'}
                                })
    print(f'saved onnx model: {onnx_name}')


if __name__ == "__main__":
    base_net = Centerface(64)
    base_net.eval()
    convert_to_onnx(base_net, 'models/my_ctf.onnx')




