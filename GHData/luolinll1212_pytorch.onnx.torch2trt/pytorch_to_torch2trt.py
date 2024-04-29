# -*- coding: utf-8 -*-  
import torch
from PIL import Image
import numpy as np
from pytorch_to_torch2trt import torch2trt
import time

from models.net import Resnet50
from config import config as cfg

import warnings

warnings.filterwarnings("ignore")

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input = torch.randn(1,3,cfg.size, cfg.size).float().to(device)

def model_run(model, description):
    t_time = []
    for x in range(100):
        img = np.random.rand(1, 32, 32, 3)
        input = torch.from_numpy(img).permute(0,3,1,2).float().to(device)
        t1 = time.time()
        out = model(input)
        t2 = time.time()
        t_time.append(t2-t1)
    print(description, "take", np.mean(np.asarray(t_time)))

"""
# pytorch fp32, fp16
"""
# pytorch fp32
model_t_32 = Resnet50(num_classes=len(cfg.classes)).eval().to(device)
model_t_32.load_state_dict(torch.load(cfg.pt))
model_run(model_t_32, "Pytorch fp32 inference")

# pytorch fp16
model_t_16 = Resnet50(num_classes=len(cfg.classes)).eval().to(device)
from src.fp16util import network_to_half
model_t_16 = network_to_half(model_t_16)
model_t_16.load_state_dict(torch.load(cfg.pt_fp16))  # fp16
model_run(model_t_16, "Pytorch fp16 inference")

"""
# torch2trt fp32, fp16
"""
img = torch.randn(1, 3, cfg.size, cfg.size).to(device)
# tensorrt fp32
model_trt_32 = torch2trt(model_t_32, [img], max_batch_size=256)
t1 = time.time()
out = model_trt_32(img)
t2 = time.time()
print("Tensorrt fp32 inference", (t2-t1))


# tensorrt fp16
img = torch.randn(1, 3, cfg.size, cfg.size).half().to(device)
model_trt_16 = torch2trt(model_t_16, [img], fp16_mode=True, max_batch_size=256)
t1 = time.time()
out = model_trt_16(img)
t2 = time.time()
print("Tensorrt fp16 inference", (t2-t1))

