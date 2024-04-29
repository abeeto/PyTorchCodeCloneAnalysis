import torch.nn as nn


class Conv1x1_BN(nn.Module):
    """
    1x1 Convolution with Batch Normalization for BasicBlock and BottleneckBlock
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1_BN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.net = self.get_network()

    def get_network(self):
        """
        returns the structure of the block
        """
        layers = []
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                stride=self.stride,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        return self.net(x)


class Conv3x3_BN(nn.Module):
    """
    3x3 Convolution with Batch Normalization for BasicBlock and BottleneckBlock
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3x3_BN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.net = self.get_network()

    def get_network(self):
        """
        returns the structure of the block
        """
        layers = []
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        return self.net(x)


class BasicBlock(nn.Module):
    """
    Basic residual block
    """

    expand = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.conv1 = self.get_conv1()
        self.conv2 = self.get_conv2()
        self.relu = nn.ReLU(inplace=True)

    def get_conv1(self):
        """
        returns the first convolution in the basic residual block
        """
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 stride=self.stride))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv2(self):
        """
        returns the second convolution in the basic residual block
        """
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv1(x)
        y = self.conv2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += x
        y = self.relu(y)

        return y


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block
    """
    expand = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.conv1 = self.get_conv1()
        self.conv2 = self.get_conv2()
        self.conv3 = self.get_conv3()
        self.ReLU = nn.ReLU(inplace=True)

    def get_conv1(self):
        """
        returns the first convolution in the bottleneck residual block
        """
        layers = []

        layers.append(Conv1x1_BN(in_channels=self.in_channels,
                                 out_channels=self.out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv2(self):
        """
        returns the second convolution in the bottleneck residual block
        """
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 stride=self.stride))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv3(self):
        """
        returns the first convolution in the bottleneck residual block
        """
        layers = []

        layers.append(Conv1x1_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels *
                                 BottleneckBlock.expand))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += x
        y = self.ReLU(y)

        return y


"""
different configurations of ResNet
"""

configs = {
    '18': [BasicBlock, 2, 2, 2, 2],
    '34': [BasicBlock, 3, 4, 6, 3],
    '50': [BottleneckBlock, 3, 4, 6, 3],
    '101': [BottleneckBlock, 3, 4, 23, 3],
    '152': [BottleneckBlock, 3, 8, 36, 3]
}


class ResNet(nn.Module):

    """ResNet Architecture"""

    def __init__(self, config, channels, class_count):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.block = configs[config][0]
        self.layer_count = configs[config][1:]
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

        self.init_weights()

    def get_conv_net(self):
        """
        returns the convolutional layers of the network
        """
        layers = []
        in_channels = self.channels

        # conv1
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=64,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(self.make_layer(block=self.block,
                                      out_channels=64,
                                      count=self.layer_count[0]))
        layers.append(self.make_layer(block=self.block,
                                      out_channels=128,
                                      count=self.layer_count[1],
                                      stride=2))
        layers.append(self.make_layer(block=self.block,
                                      out_channels=256,
                                      count=self.layer_count[2],
                                      stride=2))
        layers.append(self.make_layer(block=self.block,
                                      out_channels=512,
                                      count=self.layer_count[3],
                                      stride=2))
        layers.append(nn.AvgPool2d(kernel_size=7, stride=1))

        return nn.Sequential(*layers)

    def make_layer(self, block, out_channels, count, stride=1):
        """
        returns layers based on the type of block
        """
        downsample = None
        if (stride != 1 or self.in_channels != out_channels * block.expand):
            downsample = Conv1x1_BN(in_channels=self.in_channels,
                                    out_channels=out_channels * block.expand,
                                    stride=stride)

        layers = []
        layers.append(block(in_channels=self.in_channels,
                            out_channels=out_channels,
                            stride=stride,
                            downsample=downsample))
        self.in_channels = out_channels * block.expand
        for i in range(1, count):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def get_fc_net(self):
        """
        returns the fully connected layers of the network
        """
        return nn.Linear(512 * self.block.expand, self.class_count)

    def init_weights(self):
        """
        initializes weights for each layer
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        feed forward
        """
        y = self.conv_net(x)
        y = y.view(-1, y.size(1) * y.size(2) * y.size(3))
        y = self.fc_net(y)
        return y
