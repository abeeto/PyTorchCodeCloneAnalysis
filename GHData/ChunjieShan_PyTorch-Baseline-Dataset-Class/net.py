#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleConv3Net(nn.Module):
    def __init__(self):
        super(SimpleConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 17 * 17, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("X shape: ", x.shape)
        x = x.view(-1, 128 * 17 * 17)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(1, 3, 150, 150)
    x = torch.Tensor(x)
    model = SimpleConv3Net()
    output = model(x)
    print(output)
