import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class basic_block(nn.Module):
    '''定义了带实线部分的残差块'''
    def __init__(self, in_channels, out_channels):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x+y)


class basic_block2(nn.Module):
    '''定义了带虚线部分的残差块'''
    '''虚线的的Connection是因为x和Fx的通道数不同所以得变为'''
    def __init__(self, in_channels, out_channels):
        super(basic_block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)  # 虚线
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        z = self.bn3(self.conv3(x))
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(z+y)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.reslayers = OrderedDict()

        self.reslayers['1'] = basic_block(64, 64)
        self.reslayers['2'] = basic_block(64, 64)
        self.reslayers['3'] = basic_block2(64, 128)
        self.reslayers['4'] = basic_block(128, 128)
        self.reslayers['5'] = basic_block2(128, 256)
        self.reslayers['6'] = basic_block(256, 256)
        self.reslayers['7'] = basic_block2(256, 512)
        self.reslayers['8'] = basic_block(512, 512)

        self.avgp1 = nn.AvgPool2d(7)
        self.outlayer = nn.Linear(512, 10)  # 10类

    def forward(self, x):
        insize = x.shape[0]
        x = self.conv1(x)
        # 残差网络部分的前向传播
        for layer in self.reslayers.values():
            x = layer(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(insize, -1)
        x = self.outlayer(x)

        return x
