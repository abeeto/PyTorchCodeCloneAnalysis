import torch
import os
# import torch.nn as nn
# from models.base_model import BaseModel
from torchvision_my import models as BaseModel
import onnx
from pathlib import Path

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    # f = file.with_suffix('.onnx')
    f =  file.replace(os.path.splitext(file)[1] ,'.onnx')

    torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=not train,
                        input_names=['input'],
                        output_names=['output'],
                        # dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                        #             'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                        #             } if dynamic else None
                        dynamic_axes={'input' : {0 : 'batch_size',1:'channel'},
                                'output' : {0 : 'batch_size',1:'channel'}})

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # Simplify
    if simplify:
        try:
            import onnxsim
            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(im.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')
    print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f
 

if __name__=="__main__":
    # model_name ="mobilenetv3"
    num_classes = 3
    device ='cpu'
    im = torch.zeros(1, 3, 224,224)
    # model = BaseModel(name=model_name, num_classes=num_classes)
    model = BaseModel.resnet18(num_classes=num_classes).to(device)
    model_path = r"Z:\code\egg_products\dirt_egg_recognition\CV\Image_Classification\weights\mobilenetv3.pth"
    model.load_state_dict(torch.load(model_path))
    opset = 12
    # model.eval()
    # onnx_path = model_path.with_suffix('.onnx')
    export_onnx(model,im,model_path,opset =opset,train=False,dynamic =False,simplify =True,)