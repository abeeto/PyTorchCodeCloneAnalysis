#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleConv3Net(torch.nn.Module):
    def __init__(self):
        super(SimpleConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 96, 1, 1)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 256, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 128, 3, 2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 96, 1, 1)
        self.bn6 = nn.BatchNorm2d(96)
        self.conv7 = nn.Conv2d(96, 128, 1, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        # print("X shape: ", x.shape)
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x


if __name__ == "__main__":
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 3, 48, 48))
    model = SimpleConv3Net()
    y = model(x)
    print(y)
