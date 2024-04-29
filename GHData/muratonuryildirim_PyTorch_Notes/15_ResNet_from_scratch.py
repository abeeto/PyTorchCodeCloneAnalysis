import torch
import torch.nn as nn


class base_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, feature_maps, stride = 1, identity_downsample=None):
        super(base_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_maps, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_maps)
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            residual = self.identity_downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class bottleneck_block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, feature_maps, stride=1, identity_downsample=None):
        super(bottleneck_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_maps, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_maps)
        self.conv3 = nn.Conv2d(feature_maps, feature_maps * self.expansion, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(feature_maps * self.expansion)
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            residual = self.identity_downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, num_layers[0], feature_maps=64, stride=1)
        self.layer2 = self._make_layer(block, num_layers[1], feature_maps=128, stride=2)
        self.layer3 = self._make_layer(block, num_layers[2], feature_maps=256, stride=2)
        self.layer4 = self._make_layer(block, num_layers[3], feature_maps=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, feature_maps, stride):
        identity_downsample = None
        layers = []
        # if we halve the input space with stride=2 or change the number of channels
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead.
        # otherwise no adaptation will be needed since the shapes are matching.
        if stride != 1 or self.in_channels != feature_maps * block.expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,
                                                          feature_maps * block.expansion,
                                                          kernel_size=1,
                                                          stride=stride,
                                                          bias=False),
                                                nn.BatchNorm2d(feature_maps * block.expansion))

        layers.append(block(self.in_channels, feature_maps, stride, identity_downsample))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = feature_maps * block.expansion

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, feature_maps))
            self.in_channels = feature_maps * block.expansion

        return nn.Sequential(*layers)


def ResNet18(img_channel=3, num_classes=1000):
    return ResNet(base_block, [2, 2, 2, 2], img_channel, num_classes)


def ResNet34(img_channel=3, num_classes=1000):
    return ResNet(base_block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(bottleneck_block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(bottleneck_block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(bottleneck_block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    x = torch.randn(4, 3, 224, 224)
    model = ResNet18(img_channel=3, num_classes=1000)
    print(model(x).shape)


test()

