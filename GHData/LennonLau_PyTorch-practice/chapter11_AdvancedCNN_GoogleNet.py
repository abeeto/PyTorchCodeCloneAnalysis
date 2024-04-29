import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1_1 = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.branch1_2 = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch2 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2, stride=1)

        self.branch4_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch4_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=1)
        self.branch4_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x1 = self.branch1_2(self.branch1_1(x))

        x2 = self.branch2(x)

        x3 = self.branch3_2(self.branch3_1(x))

        x4 = self.branch4_3(self.branch4_2(self.branch4_1(x)))

        outputs = [x1, x2, x3, x4]
        return torch.cat(outputs, dim=1)


class GoogleNet(torch.nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = self.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x