# self modified 3D U-Net
# with padding=1
# with single output segmentation map

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        # level0, commented: bias=False
        self.ec0 = self.conv(in_channels=self.in_channel, out_channels=32, batch_norm=True)
        self.ec1 = self.conv(in_channels=32, out_channels=64, batch_norm=True)
        # level1
        self.ec2 = self.conv(64, 64, batch_norm=True)
        self.ec3 = self.conv(64, 128, batch_norm=True)
        # level2
        self.ec4 = self.conv(128, 128, batch_norm=True)
        self.ec5 = self.conv(128, 256, batch_norm=True)
        # level3
        self.ec6 = self.conv(256, 256, batch_norm=True)
        self.ec7 = self.conv(256, 512, batch_norm=True)

        self.pool0 = nn.MaxPool3d(kernel_size=2)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        # commented:bias=False
        self.up2 = self.up_conv(in_channels=512, out_channels=512)
        self.up1 = self.up_conv(in_channels=256, out_channels=256)
        self.up0 = self.up_conv(in_channels=128, out_channels=128)

        # level2
        self.dc6 = self.conv(256 + 512, 256, batch_norm=True)
        self.dc5 = self.conv(256, 256, batch_norm=True)

        # level1
        self.dc4 = self.conv(128 + 256, 128, batch_norm=True)
        self.dc3 = self.conv(128, 128, batch_norm=True)

        # level0
        self.dc2 = self.conv(64 + 128, 64, batch_norm=True)
        self.dc1 = self.conv(64, 64, batch_norm=True)
        self.dc0 = self.conv(64, n_classes, batch_norm=False)  # No batch norm

    @staticmethod
    def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
             bias=True, batch_norm=False):
        if batch_norm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias),
                nn.BatchNorm3d(out_channels),  # Change to 3D!
                nn.ReLU())  # PReLU!
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias),
                nn.ReLU())
        return layer

    @staticmethod
    def up_conv(in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True, dilation=1):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, bias=bias, dilation=dilation),
            nn.ReLU())
        return layer

    @staticmethod
    def center_crop(in_map, target_map):
        batch_size, n_channels, depth, height, width = in_map.size()
        target_batch_size, target_n_channels, target_depth, target_height, target_width = target_map.size()
        d = (depth - target_depth) // 2
        h = (height - target_height) // 2
        w = (width - target_width) // 2
        return in_map[:, :, d:(d+target_depth), h:(h+target_height), w:(w+target_width)]

    def forward(self, in_data):
        channel_dimension = 1  # check it
        map0 = self.ec0(in_data)
        concat0 = self.ec1(map0)

        map1 = self.pool0(concat0)
        map2 = self.ec2(map1)
        concat1 = self.ec3(map2)  # del

        map3 = self.pool1(concat1)
        map4 = self.ec4(map3)
        concat2 = self.ec5(map4)

        map5 = self.pool2(concat2)
        map6 = self.ec6(map5)
        out3 = self.ec7(map6)

        up_map6 = self.up2(out3)
        concat2_crop = self.center_crop(concat2, up_map6)
        up_map6 = torch.cat((up_map6, concat2_crop), channel_dimension)
        up_map5 = self.dc6(up_map6)
        out2 = self.dc5(up_map5)

        up_map4 = self.up1(out2)
        concat1_crop = self.center_crop(concat1, up_map4)
        up_map4 = torch.cat((up_map4, concat1_crop), channel_dimension)
        up_map3 = self.dc4(up_map4)
        out1 = self.dc3(up_map3)

        up_map2 = self.up0(out1)
        concat0_crop = self.center_crop(concat0, up_map2)
        up_map2 = torch.cat((up_map2, concat0_crop), channel_dimension)
        up_map1 = self.dc2(up_map2)
        up_map0 = self.dc1(up_map1)
        out0 = self.dc0(up_map0)
        return out0  # out1, out2, out3


if __name__ == '__main__':
    unet = UNet3D(in_channel=1, n_classes=3)  # strike
    print(unet)
