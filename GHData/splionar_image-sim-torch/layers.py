import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class DoubleConv_LeftU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == out_channels:
            mid_channels = in_channels
            out_channels = mid_channels
        else:
            mid_channels = in_channels * 2
            out_channels = mid_channels
        self.double_conv_left = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_left(x)

class DoubleConv_RightU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == out_channels:
            mid_channels = in_channels
            out_channels = mid_channels
        else:
            mid_channels = int(in_channels/2)
            out_channels = mid_channels
        self.double_conv_right = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_right(x)


class DownSampling_DoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_LeftU(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Resnet Blocks
class Unet(nn.Module):
    ''' U-net class.
    Args:
        d_dim: depth of features
    '''

    def __init__(self, d_dim):
        super().__init__()
        # Submodules
        self.convL0 = nn.Conv2d(3, d_dim, kernel_size=3, padding=1)
        self.DoubleConv2dL0 = DoubleConv_LeftU(d_dim, d_dim)
        self.Down1 = DownSampling_DoubleConv(d_dim, d_dim * 2)
        self.Down2 = DownSampling_DoubleConv(d_dim * 2, d_dim * 4)
        self.Down3 = DownSampling_DoubleConv(d_dim * 4, d_dim * 8)
        self.Down4 = DownSampling_DoubleConv(d_dim * 8, d_dim * 16)
        self.Up4 = nn.ConvTranspose2d(d_dim * 16, d_dim * 8, kernel_size=2, stride=2)
        self.DoubleConv2dR3 = DoubleConv_RightU(d_dim * 16, d_dim * 8)
        self.Up3 = nn.ConvTranspose2d(d_dim * 8, d_dim * 4, kernel_size=2, stride=2)
        self.DoubleConv2dR2 = DoubleConv_RightU(d_dim * 8, d_dim * 4)
        self.Up2 = nn.ConvTranspose2d(d_dim * 4, d_dim * 2, kernel_size=2, stride=2)
        self.DoubleConv2dR1 = DoubleConv_RightU(d_dim * 4, d_dim * 2)
        self.Up1 = nn.ConvTranspose2d(d_dim * 2, d_dim, kernel_size=2, stride=2)
        self.DoubleConv2dR0 = DoubleConv_RightU(d_dim * 2, d_dim)
        self.convR0 = nn.Conv2d(d_dim, 3, kernel_size=3, padding=1)
        
        self.actvn = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        net0 = self.actvn(self.convL0(x))
        net0 = self.pool(net0)
        net0 = self.DoubleConv2dL0(net0)
        net1 = self.Down1(net0)
        net2 = self.Down2(net1)
        net3 = self.Down3(net2)
        net = self.Down4(net3)

        net = self.Up4(net)
        net = torch.cat([net3, net], dim=1)
        net = self.DoubleConv2dR3(net)

        net = self.Up3(net)
        net = torch.cat([net2, net], dim=1)
        net = self.DoubleConv2dR2(net)

        net = self.Up2(net)
        net = torch.cat([net1, net], dim=1)
        net = self.DoubleConv2dR1(net)

        net = self.Up1(net)
        net = torch.cat([net0, net], dim=1)
        net = self.DoubleConv2dR0(net)
        net = self.convR0(net)

        return net
