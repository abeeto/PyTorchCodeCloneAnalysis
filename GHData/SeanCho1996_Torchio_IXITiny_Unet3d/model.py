import torch
import torch.nn as nn
from collections import defaultdict


down_feature = defaultdict(list)
filter_list = [i for i in range(6, 9)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = self.elu(x)
        return x


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class down_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down_sampling, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channel, out_channel),
            ConvBlock(out_channel, out_channel)
        )
        self.pool = nn.MaxPool3d(2)


    def forward(self, in_feat):
        x = self.conv(in_feat)
        down_feature[in_feat.device.index].append(x)
        x = self.pool(x)

        return x


class up_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(up_sampling, self).__init__()
        self.up_conv = ConvTranspose(in_channel, out_channel)
        self.elu_conv = nn.Sequential(
            ConvBlock(in_channel, out_channel),
            ConvBlock(out_channel, out_channel)
        )


    def forward(self, in_feat):
        x = self.up_conv(in_feat)
        down_map = down_feature[in_feat.device.index].pop()
        x = torch.cat([x, down_map], dim=1)
        x = self.elu_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.input_conv = down_sampling(1, 64)
        self.down_list = [down_sampling(2 ** i, 2 ** (i + 1)) for i in filter_list]
        self.down = nn.Sequential(*self.down_list)

        self.last_layer = nn.Sequential(
            ConvBlock(2 ** (filter_list[-1] + 1), 2 ** (filter_list[-1] + 2)),
            ConvBlock(2 ** (filter_list[-1] + 2), 2 ** (filter_list[-1] + 2))
        )

        self.up_init = up_sampling(2 ** (filter_list[-1] + 2), 2 ** (filter_list[-1] + 1))
        self.up_list = [up_sampling(2 ** (i + 1), 2 ** i) for i in filter_list[::-1]]
        self.up = nn.Sequential(*self.up_list)

        self.output = nn.Conv3d(64, num_classes, 1)
        # self.classifier = nn.Softmax()
        


    def forward(self, in_feat):
        x = self.input_conv(in_feat)
        x = self.down(x)
        x = self.last_layer(x)
        x = self.up_init(x)
        x = self.up(x)
        x = self.output(x)

        return x