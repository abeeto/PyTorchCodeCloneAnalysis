# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 16:45
import torch
from torch import nn
from torch.nn import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = nn.Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2)
        )
        self.linear_layer = nn.Sequential(
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x


# 测试model的正确性
if __name__ == '__main__':
    model = Model()
    input = torch.ones(64, 3, 32, 32)
    output = model(input)
    print(output.shape)
