import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet32(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        super(ResNet32, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=1, bias=False)  # 16,16,64
        self.res1 = nn.Sequential(ResidualBlock(64, 64, False), ResidualBlock(64, 128, True))  # 8,8,128
        self.res2 = nn.Sequential(ResidualBlock(128, 128, False), ResidualBlock(128, 256, True))  # 4,4,256
        self.res3 = nn.Sequential(ResidualBlock(256, 256, False), ResidualBlock(256, 512, True))  # 2,2,512
        self.res4 = nn.Sequential(ResidualBlock(512, 512, False), ResidualBlock(512, 1024, True))  # 1,1,1024
        self.flat = nn.Linear(1024, 100)
        self.logit = nn.Linear(100, num_classes)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)), 2, 2, 1)
        y = self.res4(self.res3(self.res2(self.res1(y))))
        y = y.view(-1, 1024)
        y = self.logit(self.flat(y))

        return y


class ResNet224(nn.Module):
    def __init__(self, channels, num_classes):
        super(ResNet224, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=1, padding=1, bias=False)  # 112,112,64
        self.res1 = nn.Sequential(ResidualBlock(64, 64, False), ResidualBlock(64, 128, True))  # 56,56,128
        self.res2 = nn.Sequential(ResidualBlock(128, 128, False), ResidualBlock(128, 256, True))  # 28,28,256
        self.res3 = nn.Sequential(ResidualBlock(256, 256, False), ResidualBlock(256, 512, True))  # 14,14,512
        self.res4 = nn.Sequential(ResidualBlock(512, 512, False), ResidualBlock(512, 1024, True))  # 7,7,1024
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)  # 1,1,1024
        self.flat = nn.Linear(1024, 500)
        self.logit = nn.Linear(500, num_classes)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)), 2, 2, 1)
        y = self.res4(self.res3(self.res2(self.res1(y))))
        y = self.avg_pool(y)
        y = y.view(-1, 1024)
        y = self.logit(self.flat(y))

        return y


class ResNet448(nn.Module):
    def __init__(self, channels, num_classes):
        super(ResNet448, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=1, padding=1, bias=False)  # 224,224,64
        self.res1 = nn.Sequential(ResidualBlock(64, 64, False), ResidualBlock(64, 128, True))  # 112,112,128
        self.res2 = nn.Sequential(ResidualBlock(128, 128, False), ResidualBlock(128, 256, True))  # 56,56,256
        self.res3 = nn.Sequential(ResidualBlock(256, 256, False), ResidualBlock(256, 512, True))  # 28,28,512
        self.res4 = nn.Sequential(ResidualBlock(512, 512, False), ResidualBlock(512, 1024, True))  # 14,14,1024
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)  # 2,2,1024
        self.flat = nn.Linear(1024 * 4, 1000)
        self.logit = nn.Linear(1000, num_classes)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        y = self.res4(self.res3(self.res2(self.res1(y))))
        y = self.avg_pool(y)
        y = y.view(-1, 1024 * 4)
        y = self.logit(self.flat(y))

        return y


class ResidualBlock(nn.Module):
    """
    Description
        Basic residual block from 'Deep Residual Learning for Image Recognition (2015)'
        Used full pre-activation from 'Identity Mappings in Deep Residual Networks (2016)'
    """
    def __init__(self, in_dim, out_dim, downsample, bias=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_dim)

        if self.downsample:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0, bias=bias)
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias)
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        x_proj = self.conv_proj(x)

        return x_proj + y


class BottleneckBlock(nn.Module):
    """
    Description
        Bottleneck block for deep layers from 'Deep Residual Learning for Image Recognition (2015)'
        Used full pre-activation from 'Identity Mappings in Deep Residual Networks (2016)'
        Because bottleneck shrinks channels to n/4, there must be at least 4 input channels
    """
    def __init__(self, in_dim, out_dim, downsample, bias=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(in_dim // 4)

        if self.downsample:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0, bias=bias)
            self.conv2 = nn.Conv2d(in_dim // 4, in_dim // 4, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias)
            self.conv2 = nn.Conv2d(in_dim // 4, in_dim // 4, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bn3 = nn.BatchNorm2d(in_dim // 4)
        self.conv3 = nn.Conv2d(in_dim // 4, out_dim, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        y = self.conv3(F.relu(self.bn3(y)))
        x_proj = self.conv_proj(x)

        return x_proj + y


# model testing
# resnet448 = ResNet448(10, 10)
# feed448 = resnet448(torch.randn(10, 10, 448, 448))
# print(feed448.size())
#
# resnet224 = ResNet224(1, 10)
# feed224 = resnet224(torch.randn(1, 1, 224, 224))
# print(feed224.size())
