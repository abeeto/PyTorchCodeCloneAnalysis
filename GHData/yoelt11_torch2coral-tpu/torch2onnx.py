import torch
import sys
sys.path.insert(0,'./torch-model/')
from model import Model
import onnx

if __name__ == '__main__':
    
    # create dummy input
    B, H, W, C = 20, 64, 256, 3
    X = torch.full((B, H, W, C),1.0, dtype=torch.float32, requires_grad=False).cpu().detach()
   
    # load torch model
    torch_weights = torch.load('./torch-model/weights/weights.pth')
    torch_model = Model(B, H, W, C)
    torch_model.load_state_dict(torch_weights)
    torch_model.eval()

    # export to onnx
    torch.onnx.export(torch_model,                  # model
                     X,                             # dummy input
                     "onnx-model/onnx_model.onnx",  # save path
                     export_params=True,             # store the trained weights inside file model
                     opset_version=16,              # onnx version model
                     do_constant_folding=False,      # constant folding optimization
                     input_names = ['input'],       # models inputs names
                     output_names = ['output'],
                     dynamic_axes={'input': {0: 'batch_size'}, # variable length
                                    'output': {0: 'batch_size'}}
                     )
    # onnx_model = onnx.load("onnx-model/onnx_model.onnx")
    # onnx.checker.check_model(onnx_model)
