import torch
import torch.nn as nn

from torch import Tensor


def conv1x1(in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     bias=False)


def conv3x3(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     bias=False, padding=padding)


class BasicBlock(nn.Module):
    block_expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block_conv = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            conv3x3(in_channels=out_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.shortcut_conv = nn.Sequential(
            conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x_identity = x

        x = self.block_conv(x)

        if x.shape != x_identity.shape:
            x_identity = self.shortcut_conv(x_identity)

        x += x_identity
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    block_expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        block_out_channels = out_channels * 4
        self.block_conv = nn.Sequential(
            conv1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            conv3x3(in_channels=out_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            conv1x1(in_channels=out_channels, out_channels=block_out_channels),
            nn.BatchNorm2d(num_features=block_out_channels),
        )
        self.shortcut_conv = nn.Sequential(
            conv1x1(in_channels=in_channels, out_channels=block_out_channels, stride=stride),
            nn.BatchNorm2d(block_out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x_identity = x

        x = self.block_conv(x)

        if x.shape != x_identity.shape:
            x_identity = self.shortcut_conv(x_identity)

        x += x_identity
        x = self.relu(x)
        return x


def create_block(block, num_repeats: int, channels: int, init_stride: int = 2):
    layers = list()
    if init_stride == 1:
        layers.append(block(in_channels=channels, out_channels=channels))
    else:
        in_channels = channels * 2
        if block.block_expansion == 1:
            in_channels = channels // 2
        layers.append(block(in_channels=in_channels, out_channels=channels, stride=init_stride))

    in_channels = channels * block.block_expansion
    for _ in range(num_repeats - 1):
        layers.append(block(in_channels=in_channels, out_channels=channels))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
    PyTorch implementation of ResNet model.
    Paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, num_classes: int, in_channels: int, num_block_repeats: list, block):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = create_block(block, num_block_repeats[0], 64, init_stride=1)
        self.conv3_x = create_block(block, num_block_repeats[1], 128)
        self.conv4_x = create_block(block, num_block_repeats[2], 256)
        self.conv5_x = create_block(block, num_block_repeats[3], 512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 512
        if block.block_expansion == 4:
            in_features = 2048
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def resnet50(num_classes: int = 1000, in_channels: int = 3):
    return ResNet(num_classes, in_channels, [3, 4, 6, 3], BottleNeck)


def resnet101(num_classes: int = 1000, in_channels: int = 3):
    return ResNet(num_classes, in_channels, [3, 4, 23, 3], BottleNeck)


def resnet152(num_classes: int = 1000, in_channels: int = 3):
    return ResNet(num_classes, in_channels, [3, 8, 36, 3], BottleNeck)


def resnet18(num_classes: int = 1000, in_channels: int = 3):
    return ResNet(num_classes, in_channels, [2, 2, 2, 2], BasicBlock)


def resnet34(num_classes: int = 1000, in_channels: int = 3):
    return ResNet(num_classes, in_channels, [3, 4, 6, 3], BasicBlock)


def main():
    model = resnet152()
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    output = model(x)
    assert output.shape == (1, 1000)


if __name__ == '__main__':
    main()
