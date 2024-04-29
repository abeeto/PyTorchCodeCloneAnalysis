from abc import ABC
import torch.nn as nn


class ResBlock(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.sequential(x) + self.batchNorm(self.conv(x))


class ResNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.modules.flatten.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
