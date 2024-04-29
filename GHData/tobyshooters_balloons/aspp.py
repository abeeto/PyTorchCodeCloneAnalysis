import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference: github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py

class ImagePool(nn.Module):
    def __init__(self, inc, intc):
        super(ImagePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(inc, intc, kernel_size=1)
        self.bn = nn.BatchNorm2d(intc)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.pool(x)
        x = self.relu(self.bn(self.conv(x)))
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self, inc, intc):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(inc, intc, 1, 1, 0),
            nn.BatchNorm2d(intc),
            nn.ReLU()
        )
        self.aspp6 = nn.Sequential(
            nn.Conv2d(inc, intc, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(intc),
            nn.ReLU()
        )
        self.aspp12 = nn.Sequential(
            nn.Conv2d(inc, intc, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(intc),
            nn.ReLU()
        )
        self.aspp18 = nn.Sequential(
            nn.Conv2d(inc, intc, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(intc),
            nn.ReLU()
        )
        self.pool = ImagePool(inc, intc)

        self.post = nn.Sequential(
            nn.Conv2d(5 * intc, intc, 1),
            nn.BatchNorm2d(intc),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp6(x)
        x3 = self.aspp12(x)
        x4 = self.aspp18(x)
        x5 = self.pool(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.post(x)
        return x
