import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.linear = nn.Linear(64, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x