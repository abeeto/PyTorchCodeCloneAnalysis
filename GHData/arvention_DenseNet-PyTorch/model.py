import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """
    Dense layer
    """

    def __init__(self,
                 in_channels,
                 expand_factor=4,
                 growth_rate=32):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bottleneck_size = expand_factor * growth_rate

        self.conv1x1 = self.get_conv1x1()
        self.conv3x3 = self.get_conv3x3()

    def get_conv1x1(self):
        """
        returns a stack of Batch Normalization, ReLU, and
        1x1 Convolution layers
        """
        layers = []
        layers.append(nn.BatchNorm2d(num_features=self.in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.bottleneck_size,
                                kernel_size=1,
                                stride=1,
                                bias=False))

        return nn.Sequential(*layers)

    def get_conv3x3(self):
        """
        returns a stack of Batch Normalization, ReLU, and
        3x3 Convolutional layers
        """
        layers = []
        layers.append(nn.BatchNorm2d(num_features=self.bottleneck_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.bottleneck_size,
                                out_channels=self.growth_rate,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv1x1(x)
        y = self.conv3x3(y)

        y = torch.cat([x, y], 1)

        return y


class DenseBlock(nn.Module):
    """
    Dense block
    """

    def __init__(self,
                 in_channels,
                 num_layers,
                 expand_factor=4,
                 growth_rate=32):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.expand_factor = expand_factor
        self.growth_rate = growth_rate

        self.net = self.get_network()

    def get_network(self):
        """
        return num_layers dense layers
        """
        layers = []

        for i in range(self.num_layers):
            in_channels = self.in_channels + i * self.growth_rate
            layers.append(DenseLayer(in_channels=in_channels,
                                     expand_factor=self.expand_factor,
                                     growth_rate=self.growth_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        return self.net(x)


class TransitionBlock(nn.Module):
    """
    Transition block
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = self.get_network()

    def get_network(self):
        """
        returns the structure of the block
        """
        layers = []

        layers.append(nn.BatchNorm2d(num_features=self.in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                stride=1,
                                bias=False))
        layers.append(nn.AvgPool2d(kernel_size=2,
                                   stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward pass
        """
        return self.net(x)


"""
different configurations of DenseNet
"""

configs = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '264': [6, 12, 64, 48]
}


class DenseNet(nn.Module):

    """DenseNet Architecture"""

    def __init__(self,
                 config,
                 channels,
                 class_count,
                 num_features=64,
                 compress_factor=2,
                 expand_factor=4,
                 growth_rate=32):
        super(DenseNet, self).__init__()
        self.config = configs[config]
        self.channels = channels
        self.class_count = class_count

        self.num_features = num_features
        self.compress_factor = compress_factor
        self.expand_factor = expand_factor
        self.growth_rate = growth_rate

        self.conv_net = self.get_conv_network()
        self.fc_net = self.get_fc_net()

        self.init_weights()

    def get_conv_network(self):
        """
        returns the convolutional layers of the network
        """
        layers = []

        layers.append(nn.Conv2d(in_channels=self.channels,
                                out_channels=self.num_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=self.num_features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3,
                                   stride=2,
                                   padding=1))

        for i, num_layers in enumerate(self.config):
            layers.append(DenseBlock(in_channels=self.num_features,
                                     num_layers=num_layers,
                                     expand_factor=self.expand_factor,
                                     growth_rate=self.growth_rate))

            self.num_features += num_layers * self.growth_rate

            if i != len(self.config) - 1:
                out_channels = self.num_features // self.compress_factor
                layers.append(TransitionBlock(in_channels=self.num_features,
                                              out_channels=out_channels))

                self.num_features = out_channels

        layers.append(nn.BatchNorm2d(num_features=self.num_features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AvgPool2d(kernel_size=7,
                                   stride=1))

        return nn.Sequential(*layers)

    def get_fc_net(self):
        """
        returns the fully connected layers of the network
        """
        return nn.Linear(in_features=self.num_features,
                         out_features=self.class_count)

    def init_weights(self):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv_net(x)
        y = y.view(-1, y.size(1) * y.size(2) * y.size(3))
        y = self.fc_net(y)
        return y
