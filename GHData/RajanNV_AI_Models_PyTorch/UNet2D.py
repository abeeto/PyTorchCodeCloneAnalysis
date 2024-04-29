import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

class EXCC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EXCC, self).__init__()
        self.ec = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU() )

    def forward(self, x):
        x = self.ec(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()
        n1 = 4
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        aa = [224, 112, 56]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d(p = 0.7)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Conv2d(filters[0], aa[0], kernel_size=3, stride=1, padding=1, bias=True)

        self.Con1 = EXCC(aa[0], aa[1])
        self.Con2 = EXCC(aa[1], aa[2])
        self.Con3 = EXCC(aa[2], out_ch)

        # self.Conv21 = nn.Conv2d(aa[0], aa[1], kernel_size=3, stride=2, padding=1, bias=True)
        # self.Conv22 = nn.Conv2d(aa[1], aa[2], kernel_size=3, stride=2, padding=1, bias=True)
        # self.Conv23 = nn.Conv2d(aa[2], out_ch, kernel_size=3, stride=1, padding=1, bias=True)

        # self.fc1 = nn.Linear(aa[0], aa[1])
        # self.fc2 = nn.Linear(aa[1], aa[2])
        # self.fc3 = nn.Linear(aa[2], out_ch)
    def forward(self, x):
        d1 = self.Conv1(x)

        d2 = self.Maxpool(d1)
        d3 = self.drop(d2)
        d4 = self.Conv2(d3)

        d5 = self.Maxpool(d4)
        d6= self.drop(d5)
        d7 = self.Conv3(d6)

        d8 = self.Maxpool(d7)
        d9  = self.drop(d8)
        d10 = self.Conv4(d9)

        d11 = self.Maxpool(d10)
        d12 = self.drop(d11)
        d13 = self.Conv5(d12)

        u15 = self.Up5(d13)
        u14 = torch.cat((d10, u15), dim=1)

        u13 = self.Up_conv5(u14)

        u12= self.Up4(u13)
        u11 = torch.cat((d7, u12), dim=1)
        u10 =   self.drop(u11)
        u9 = self.Up_conv4(u10)

        u8 = self.Up3(u9)
        u7 = torch.cat((d4, u8), dim=1)
        u6 = self.drop(u7)
        u5 = self.Up_conv3(u6)

        u4 = self.Up2(u5)
        u3 = torch.cat((d1, u4), dim=1)
        u2 = self.drop(u3)
        u1 = self.Up_conv2(u2)
        u = self.Conv(u1)
        # return  u
        #
        # ff1=self.Conv21(u)
        # ff1=self.Conv22(ff1)
        # ff1=self.Conv23(ff1)
        # return  ff1

        ff1=self.Con1(u)
        ff1= self.Maxpool(ff1)
        ff1 = self.drop(ff1)
        ff1=self.Con2(ff1)
        ff1= self.Maxpool(ff1)
        ff1 = self.drop(ff1)
        ff1=self.Con3(ff1)
        return ff1
