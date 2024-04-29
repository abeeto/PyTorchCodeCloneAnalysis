import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.conv2(x)

        return self.relu(y+x)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(self.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(self.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x
