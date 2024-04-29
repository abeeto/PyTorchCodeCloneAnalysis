import torch
import torch.nn as nn
import torch.nn.functional as F

class StupidNet(nn.Module):
    def __init__(self, n_classes):  # constructor
        super(StupidNet, self).__init__()  # parent constructor

        self.drop = nn.Dropout2d(p=0.2)

        # in_channels, out_channels, kernel_size
        self.conv0 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv0_bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, n_classes, 5, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv0_bn(self.conv0(x)))
        x = self.drop(x)
        x = self.conv1(x)
        x = F.softmax(x, dim=1)  # apply softmax along dim 1 (dim 0 is the different batches)
        return x