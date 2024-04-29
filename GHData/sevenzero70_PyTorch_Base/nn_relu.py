# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 10:02
import torch
import torchvision.datasets
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = Linear()
    def forward(self, input):
        input_1 = torch.reshape(input, (1, 1, 1, -1))
        input_2 = torch.flatten(img)        # 直接展平

for data in dataloader:
    img, target = data
    print(img.shape)

