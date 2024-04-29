# %%
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Flatten(),
            nn.Linear(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class AtrousModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AtrousModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Flatten(),
            nn.Linear(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class DepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super(DepthBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.block(x)


class DepthModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DepthModel, self).__init__()
        self.backbone = nn.Sequential(
            DepthBlock(in_channels=3, out_channels=128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            DepthBlock(in_channels=128, out_channels=256, kernel_size=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            DepthBlock(in_channels=256, out_channels=512, kernel_size=7),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Flatten(),
            nn.Linear(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# %%
# inp= BaseModel()
# print(f'base: {count_parameters(b):,}')
# a = AtrousModel()
# print(f'atrous: {count_parameters(a):,}')
# d = DepthModel()
# print(f'depth: {count_parameters(d):,}')

# b(inp)

# print(f'Base..: {count_parameters(b):,}')
# print(f'Atrous: {count_parameters(a):,}')
# print(f'Depth.: {count_parameters(d):,}')


# %%
# inp = torch.rand(1, 3, 7, 7)
# m1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, bias=False)
# m2 = nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, bias=False),
#     nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, bias=False)
# )
# m3 = DepthBlock(3, 128, bias=False)
# # m = nn.Conv2d(3, 128, 3, bias=False)
# print(count_parameters(m1))
# print(count_parameters(m3))

# # %%
