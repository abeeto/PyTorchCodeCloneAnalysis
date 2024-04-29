import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import numpy as np
import os
import segmentation_models_pytorch as smp
from haudi import AudiSegmentationDataset
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from stupidNet import *
from watoNet import *
from WeightedFocalLoss import FocalLoss
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
multi_gpu = True
conf = {
    "epochs": 150,
    "learning_rate": 0.0004,
    "momentum": 0.9,
    "batch_size": 2,
    "weight_decay": 0.0001,
    # "weight_balance": [1.03460231848, 276.939057959, 107.45530397, 124.671731118, 87.0446761785, 981.810059761],
    # "weight_balance": [1, 1, 1, 1, 1, 1],
    "weight_balance": [1, 50, 200],
    # "weight_balance": [1, 50],
    "backbone": "efficientnet-b1",
    "positional_encoding": True,
    "loss_function": "Xentropy"
}

model = smp.Unet(conf["backbone"], encoder_weights='imagenet', in_channels=5, classes=3,
                 activation='softmax2d')
state_dict = torch.load("wandb/run-20200430_154829-9hasgfy0/WATONET-10-0.9875838115676441.pth")
# create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
model = nn.DataParallel(model)
model.load_state_dict(state_dict)

model.module.encoder.set_swish(memory_efficient=False)
model.eval()
model = model.cpu()
dummy_input = torch.rand((1, 5, 960, 320)).to("cpu")
# torch.onnx.export(model.module, dummy_input, 'model.onnx')
traced_script_module = torch.jit.trace(model.module, dummy_input)
traced_script_module.save("traced_model.pt")

print(traced_script_module(dummy_input))
