import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    """Fire Module based on SqueezeNet"""

    def __init__(
        self,
        in_channels,
        squeeze_channels,
        e1x1_channels,
        e3x3_channels
    ):
        super(Fire, self).__init__()

        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.e1x1_channels = e1x1_channels
        self.e3x3_channels = e3x3_channels

        self.squeeze_layer = self.get_squeeze_layer()
        self.expand_1x1_layer = self.get_expand_1x1_layer()
        self.expand_3x3_layer = self.get_expand_3x3_layer()

    def get_squeeze_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.in_channels,
                                self.squeeze_channels,
                                kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_1x1_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e1x1_channels,
                                kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_3x3_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e3x3_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.squeeze_layer(x)
        return torch.cat([
            self.expand_1x1_layer(y),
            self.expand_3x3_layer(y)
        ], 1)


class SqueezeNet2(nn.Module):

    """SqueezeNet1.1"""

    def __init__(
        self,
        channels,
        class_count
    ):
        super(SqueezeNet2, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.features = self.get_features()
        self.classifier = self.get_classifier()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)

                else:
                    init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_features(self):
        layers = []

        # in_channels = self.channels, out_channels = 64
        # kernel_size = 3x3, stride = 2
        layers.append(nn.Conv2d(self.channels, 64, kernel_size=3, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 64, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        layers.append(Fire(64, 16, 64, 64))

        # in_channels = 128, squeeze_channels = 16
        # e1x1_channels = 64, e3x3_channels = 64 -> out_channels = 128
        layers.append(Fire(128, 16, 64, 64))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 128, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        layers.append(Fire(128, 32, 128, 128))

        # in_channels = 256, squeeze_channels = 32
        # e1x1_channels = 128, e3x3_channels = 128 -> out_channels = 256
        layers.append(Fire(256, 32, 128, 128))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        # in_channels = 256, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        layers.append(Fire(256, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 48
        # e1x1_channels = 192, e3x3_channels = 192 -> out_channels = 384
        layers.append(Fire(384, 48, 192, 192))

        # in_channels = 384, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        layers.append(Fire(384, 64, 256, 256))

        # in_channels = 512, squeeze_channels = 64
        # e1x1_channels = 256, e3x3_channels = 256 -> out_channels = 512
        layers.append(Fire(512, 64, 256, 256))

        return nn.Sequential(*layers)

    def get_classifier(self):
        layers = []

        self.final_conv = nn.Conv2d(512, self.class_count, kernel_size=1)

        layers.append(nn.Dropout())
        layers.append(self.final_conv)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AvgPool2d(13, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.features(x)
        y = self.classifier(y)
        return y.view(y.size(0), self.class_count)
