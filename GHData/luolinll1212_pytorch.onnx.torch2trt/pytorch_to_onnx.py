# -*- coding: utf-8 -*-
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from models.net import Resnet50
from config import config as cfg

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input = torch.rand((1, 3, cfg.size, cfg.size)).float().to(device)

# load the model
model = Resnet50(num_classes=len(cfg.classes)).to(device)
if not cfg.fp16:
    model.load_state_dict(torch.load(cfg.pt))  # fp32
    torch.onnx.export(model, input, cfg.onnx, verbose=True, export_params=True, opset_version=9)
else:
    from src.fp16util import network_to_half
    model = network_to_half(model)
    model.load_state_dict(torch.load(cfg.pt_fp16))  # fp16
    torch.onnx.export(model, input, cfg.onnx_fp16, verbose=True, export_params=True, opset_version=9)
