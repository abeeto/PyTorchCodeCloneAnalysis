import torch
import torch.nn as nn
import torch.nn.init as init


class ZipLayer(nn.Module):
    """Zip Module"""

    def __init__(self,
                 in_channels,
                 squeeze_channels,
                 e1x1_channels,
                 e3x3_channels):
        super(ZipLayer, self).__init__()

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
        layers.append(nn.BatchNorm2d(num_features=self.squeeze_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_1x1_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e1x1_channels,
                                kernel_size=1))
        layers.append(nn.BatchNorm2d(num_features=self.e1x1_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_expand_3x3_layer(self):
        layers = []

        layers.append(nn.Conv2d(self.squeeze_channels,
                                self.e3x3_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.BatchNorm2d(num_features=self.e3x3_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.squeeze_layer(x)
        return torch.cat([
            self.expand_1x1_layer(y),
            self.expand_3x3_layer(y)
        ], 1)


class ZipBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 num_layers,
                 compress_factor,
                 expand_factor,
                 expand_interval):
        super(ZipBlock, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.compress_factor = compress_factor
        self.expand_factor = expand_factor
        self.expand_interval = expand_interval

        self.net = self.get_network()

    def get_network(self):
        layers = []

        in_channels = self.in_channels
        for i in range(self.num_layers):
            squeeze_channels = in_channels // self.compress_factor

            out_channels = in_channels
            if (i + 1) % self.expand_interval == 0:
                out_channels *= self.expand_factor

            layers.append(ZipLayer(in_channels=in_channels,
                                   squeeze_channels=squeeze_channels,
                                   e1x1_channels=out_channels // 2,
                                   e3x3_channels=out_channels // 2))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


"""
different configurations of ZipNet
"""

configs = {
    'A': [64, [64, 4, 4, 2, 2], [256, 2, 8, 1, 1], [256, 4, 8, 2, 3]],
    'B': [64, [64, 4, 4, 2, 2], [256, 2, 8, 1, 1], [256, 4, 8, 2, 2]]
}


class ZipNet(nn.Module):

    """ZipNet"""

    def __init__(self,
                 config,
                 channels,
                 class_count):
        super(ZipNet, self).__init__()
        self.config = configs[config]
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()

    def get_conv_net(self):
        layers = []

        layers.append(nn.Conv2d(in_channels=self.channels,
                                out_channels=self.config[0],
                                kernel_size=3,
                                stride=2))
        layers.append(nn.BatchNorm2d(num_features=self.config[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,
                                   stride=2,
                                   ceil_mode=True))

        layers.append(ZipBlock(in_channels=self.config[1][0],
                               num_layers=self.config[1][1],
                               compress_factor=self.config[1][2],
                               expand_factor=self.config[1][3],
                               expand_interval=self.config[1][4]))
        layers.append(nn.MaxPool2d(kernel_size=2,
                                   stride=2,
                                   ceil_mode=True))

        layers.append(ZipBlock(in_channels=self.config[2][0],
                               num_layers=self.config[2][1],
                               compress_factor=self.config[2][2],
                               expand_factor=self.config[2][3],
                               expand_interval=self.config[2][4]))
        layers.append(nn.MaxPool2d(kernel_size=2,
                                   stride=2,
                                   ceil_mode=True))

        layers.append(ZipBlock(in_channels=self.config[3][0],
                               num_layers=self.config[3][1],
                               compress_factor=self.config[3][2],
                               expand_factor=self.config[3][3],
                               expand_interval=self.config[3][4]))

        expand_count = self.config[3][1] // self.config[3][4]
        final_channels = self.config[3][0]
        for i in range(expand_count):
            final_channels *= self.config[3][3]

        self.final_conv = nn.Conv2d(in_channels=final_channels,
                                    out_channels=self.class_count,
                                    kernel_size=1)

        layers.append(nn.Dropout())
        layers.append(self.final_conv)
        layers.append(nn.BatchNorm2d(num_features=self.class_count))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def init_weights(self):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is self.final_conv:
                    init.normal_(module.weight, mean=0.0, std=0.01)

                else:
                    init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        y = self.conv_net(x)
        pool = nn.AvgPool2d(y.size(2), stride=1)
        y = pool(y)
        return y.view(y.size(0), self.class_count)
