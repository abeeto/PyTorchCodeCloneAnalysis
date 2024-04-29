import torch
import torch.nn as nn

# UNet reference: github.com/milesial/Pytorch-UNet
# Attention reference: https://github.com/LeeJunHyun/Image_Segmentation

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, int_c=None):
        super().__init__()
        if not int_c:
            int_c = out_c
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, int_c, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(int_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_c, out_c, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Attention(nn.Module):
    def __init__(self, gate_c, layer_c, int_c):
        super(Attention, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(gate_c, int_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_c)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(layer_c, int_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_c)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_up, x_skip, visualize=False):
        g = self.Wg(x_up)
        x = self.Wx(x_skip)
        j = self.relu(g + x)
        p = self.psi(j)
        return p * x_skip, p


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    def __init__(self, up_c, skip_c, out_c, att_c=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(up_c + skip_c, out_c, (up_c + skip_c) // 2)

        if att_c:
            self.att = Attention(up_c, skip_c, att_c)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)
        if self.att:
            x_skip, _ = self.att(x_up, x_skip)
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.in_conv = nn.Conv2d(in_c, 64, 3, 1, 1)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up4 = Up(512, 512, 256, att_c=256)
        self.up3 = Up(256, 256, 128, att_c=128)
        self.up2 = Up(128, 128, 64,  att_c=64)
        self.up1 = Up(64,  64,  64,  att_c=32)
        self.out_conv = nn.Conv2d(64, out_c, 3, 1, 1)


    def forward(self, *args):
        x = torch.cat(args, dim=1)

        xi = self.in_conv(x)
        d1 = self.down1(xi)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4, d3) 
        u3 = self.up3(u4, d2)
        u2 = self.up2(u3, d1)
        u1 = self.up1(u2, xi)
        y = self.out_conv(u1)

        return torch.sigmoid(y)
