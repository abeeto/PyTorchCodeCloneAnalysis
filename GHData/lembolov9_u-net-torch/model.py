import torch
from torch import nn


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.out(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
			nn.ReLU()
)

    def forward(self, x):
        x = self.up(x)
        return x


class uNet(nn.Module):
    def __init__(self):
        super(uNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_1 = down_conv(1, 64)
        self.down_2 = down_conv(64, 128)
        self.down_3 = down_conv(128, 256)
        self.down_4 = down_conv(256, 512)
        self.down_5 = down_conv(512, 1024)

        self.de_up_5 = up(1024, 512)
        self.up_5 = down_conv(1024, 512)

        self.de_up_4 = up(512, 256)
        self.up_4 = down_conv(512, 256)

        self.de_up_3 = up(256, 128)
        self.up_3 = down_conv(256, 128)

        self.de_up_2 = up(128, 64)
        self.up_2 = down_conv(128, 64)

        self.out = out(64, 1)

    def forward(self, x):
        x1 = self.down_1(x)

        x2 = self.pool(x1)
        x2 = self.down_2(x2)

        x3 = self.pool(x2)
        x3 = self.down_3(x3)

        x4 = self.pool(x3)
        x4 = self.down_4(x4)

        x5 = self.pool(x4)
        x5 = self.down_5(x5)

        x6 = self.de_up_5(x5)
        x6 = torch.cat((x4, x6), dim=1)
        x6 = self.up_5(x6)

        x7 = self.de_up_4(x6)
        x7 = torch.cat((x3, x7), dim=1)
        x7 = self.up_4(x7)

        x8 = self.de_up_3(x7)
        x8 = torch.cat((x2, x8), dim=1)
        x8 = self.up_3(x8)

        x9 = self.de_up_2(x8)
        x9 = torch.cat((x1, x9), dim=1)
        x9 = self.up_2(x9)

        x10 = self.out(x9)

        return x10




