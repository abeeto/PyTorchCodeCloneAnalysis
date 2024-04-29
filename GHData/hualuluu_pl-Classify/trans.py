import os, cv2
import torch
import torchvision 
from PIL import Image
import numpy as np
import torch.nn as nn

model_path = '/media/worker/98f56e45-a9c8-4f4e-b3f1-08a2d16a7ec1/liliang/Project/Classify/runs/20220902/weights/version_5/checkpoints/epoch=49-val_loss=0.03-val_acc=1.00.ckpt'

model = torchvision.models.mobilenet_v2(pretrained=False)
# Update Model Structure
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.last_channel, model.last_channel // 2),
    nn.LeakyReLU(0.1),
    nn.Linear(model.last_channel // 2, 3)
)

# model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
pretrained_dict = torch.load(model_path)['state_dict']
new_pretrained_dict = {}
for k, v in pretrained_dict.items():
    if k.split('model.')[-1] in model.state_dict():
        # print(k, k.split('model.')[-1])
        new_k = k.split('model.')[-1]
        new_pretrained_dict[new_k] = v
model.load_state_dict(new_pretrained_dict)
torch.save(model.state_dict(),'model_state_dict_version5.pth')
