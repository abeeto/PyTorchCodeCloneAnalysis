import torch
import torch.onnx
import onnx
from model.lenet import LeNet

def export(model_name, ckpt_name):
  ckpt_path = ckpt_name+'.pth'
  onnx_model = ckpt_name+'.onnx'
  
# model definition
  if model_name == 'lenet':
    from model.lenet import LeNet
    model = LeNet()
  else:
    from model.modelzoo import create_model
    model, input_size = create_model(model_name, n_classes=120)

  # load weights
  ckpt = torch.load(ckpt_path)
  model.load_state_dict(ckpt['state_dict'])
  
  # evaluation mode
  model.eval()

  # create the imput placeholder for the model
  # note: we have to specify the size of a batch of input images
  if model_name == 'lenet':
    input_placeholder = torch.randn(1, 1, 28, 28)
  elif model_name == 'inception_v3':
    input_placeholder = torch.randn(1, 3, 299, 299)
  else:
    input_placeholder = torch.randn(1, 3, 224, 224)

  # export
  torch.onnx.export(model, input_placeholder, onnx_model)
  print('{} exported!'.format(onnx_model))

def print_onnx(onnx_model):
  model = onnx.load(onnx_model)
  onnx.checker.check_model(model)
  print('Contents of this model {}:'.format(onnx_model))
  print(onnx.helper.printable_graph(model.graph))

if __name__ == '__main__':
  from config import lr, model_name, ckpt_name
  export(model_name, ckpt_name)
  onnx_model = ckpt_name + '.onnx'
  print_onnx(onnx_model)

