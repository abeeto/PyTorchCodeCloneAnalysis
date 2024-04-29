# 这里使用迁移学习
import torchvision
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn

model_conv = torchvision.models.resnet18(pretrained=True)
# 在这里ConvNet作为固定特征提取器
for param in model_conv.parameters():
    param.required_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
