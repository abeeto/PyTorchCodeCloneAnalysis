import torch
import torch.nn as nn
import torch.nn.functional as funct


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(2, stride=2)
        # linear
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, n_classes)
        # activation
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        probs = funct.softmax(x, dim=1)
        return x, probs
