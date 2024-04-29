import torch
import torch.onnx
from model import DeepSpeech 

## A model class instance (class not shown)
pytorch_model = DeepSpeech(labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs ",rnn_hidden_size=128,nb_layers=2)

## Load the weights from a file (.pth usually)
state_dict = torch.load("./24-6-2019/deepspeech_final.pth")

## Load the weights now into a model net architecture defined by our class
pytorch_model.load_state_dict(state_dict["state_dict"])
#print(pytorch_model)

## Create the right input shape (e.g. for an image)
dummy_input = torch.randn(8, 1,3316,3316,device='cuda')
#dummy_input = torch.randn(8, 1, 32,32,lenghts=[3316,3316,3316,3316])

#torch.cuda.get_device_name(0)
torch.onnx.export(pytorch_model, dummy_input, "deepspeech.onnx")
