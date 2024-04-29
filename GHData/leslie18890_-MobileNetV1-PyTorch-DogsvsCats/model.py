import torch
import torch.nn as nn
import torch.nn.functional as F

class Depthwise_separable_convolution(nn.Module):
    '''DW + PW'''
    def __init__(self, in_channel, out_channel, stride=1):
        super(Depthwise_separable_convolution, self).__init__()
        #DW
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        #PW
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNetV1(nn.Module):
    # 除第一层外，后面的层均为深度可分离卷积层，cfg存储深度可分离卷积层 Fruture Maps 的 channels 和 卷积步长 padding
    cfg = [(64,1), (128,2), (128,1), (256,2), (256,1), (512,2),
           (512,1), (512,1), (512,1), (512,1), (512,1), (1024,2), (1024,1)]

    def __init__(self, num_classes=1000 ):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x[0]
            stride = x[1]
            layers.append(Depthwise_separable_convolution(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        #out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNetV1_075(nn.Module):
    # 除第一层外，后面的层均为深度可分离卷积层，cfg存储深度可分离卷积层 Fruture Maps 的 channels 和 卷积步长 padding
    cfg = [(48,1), (96,2), (96,1), (192,2), (192,1), (384,2),
           (384,1), (384,1), (384,1), (384,1), (384,1), (768,2), (768,1)]

    def __init__(self, num_classes=1000 ):
        super(MobileNetV1_075, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.layers = self._make_layers(in_channels=24)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(768, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x[0]
            stride = x[1]
            layers.append(Depthwise_separable_convolution(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        #out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNetV1_050(nn.Module):
    # 除第一层外，后面的层均为深度可分离卷积层，cfg存储深度可分离卷积层 Fruture Maps 的 channels 和 卷积步长 padding
    cfg = [(32,1), (64,2), (64,1), (128,2), (128,1), (256,2),
           (256,1), (256,1), (256,1), (256,1), (246,1), (512,2), (512,1)]

    def __init__(self, num_classes=1000 ):
        super(MobileNetV1_050, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_channels=16)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(512, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x[0]
            stride = x[1]
            layers.append(Depthwise_separable_convolution(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        #out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNetV1_025(nn.Module):
    # 除第一层外，后面的层均为深度可分离卷积层，cfg存储深度可分离卷积层 Fruture Maps 的 channels 和 卷积步长 padding
    cfg = [(8,1), (16,2), (32,1), (64,2), (64,1), (128,2),
           (128,1), (128,1), (128,1), (128,1), (128,1), (256,2), (256,1)]

    def __init__(self, num_classes=1000 ):
        super(MobileNetV1_025, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.layers = self._make_layers(in_channels=4)
        #self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(256, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x[0]
            stride = x[1]
            layers.append(Depthwise_separable_convolution(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        #out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out