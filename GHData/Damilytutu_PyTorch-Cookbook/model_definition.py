# coding: utf-8
import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision


# 卷积层
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)


# global average pooling
gap = torch.nn.AdaptiveAvgPool2d(output_size=1)


# 双线性汇合（bilinear pooling）
X = torch.reshape(N, D, H * W)                        # Assume X has shape N*D*H*W
X = torch.bmm(X, torch.transpose(X, 1, 2)) / (H * W)  # Bilinear pooling
assert X.size() == (N, D, D)
X = torch.reshape(X, (N, D * D))
X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)   # Signed-sqrt normalization
X = torch.nn.functional.normalize(X)                  # L2 normalization


# 多卡同步BN（Batch normalization）
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch


# 类似BN滑动平均
class BN(torch.nn.Module):
    def __init__(self):
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, X):
        self.running_mean += momentum * (current - self.running_mean)  # 原地操作（in_place）


# 计算模型整体参数量
from torchvision.models import resnet50
model = resnet50()
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())


# 输出模型信息，比如每层的参数量
# https://github.com/sksq96/pytorch-summary
import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

summary(vgg, (3, 224, 224))


# 模型权值初始化
# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)


# 部分层使用预训练模型
model.load_state_dict(torch.load('model,pth'), strict=False)


# 将在GPU保存的模型加载到CPU
model.load_state_dict(torch.load('model,pth', map_location='cpu'))


