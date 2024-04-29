import torch.nn.functional as F
from torch import nn


class DownSample(nn.Module):
    def __init__(self, in_channel, sc_channel):
        super(DownSample, self).__init__()
        self.rp = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channel, sc_channel, 7, padding=0)
        self.bn1 = nn.BatchNorm2d(sc_channel)
        self.conv2 = nn.Conv2d(sc_channel, sc_channel * 2, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(sc_channel * 2)
        self.conv3 = nn.Conv2d(sc_channel * 2, sc_channel * 4, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(sc_channel * 4)

    def forward(self, x):
        out = self.rp(x)
        out = F.relu(self.bn1(self.conv1(out)), True)
        out = F.relu(self.bn2(self.conv2(out)), True)
        out = F.relu(self.bn3(self.conv3(out)), True)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3)
        self.rp2 = nn.ReflectionPad2d(1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3)
        self.bn2 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        out = self.rp1(x)
        out = F.relu(self.bn1(self.conv1(out)), True)
        out = self.rp2(out)
        out = self.bn2(self.conv2(out))

        return out + x


class Upsample(nn.Module):
    def __init__(self, sc_channel, out_channel):
        super(Upsample, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(sc_channel, sc_channel // 2, 3, 2, 1, 1)
        self.bn1 = nn.BatchNorm2d(sc_channel // 2)
        self.tconv2 = nn.ConvTranspose2d(sc_channel // 2, sc_channel // 4, 3, 2, 1, 1)
        self.bn2 = nn.BatchNorm2d(sc_channel // 4)
        self.rp = nn.ReflectionPad2d(3)
        self.conv3 = nn.Conv2d(sc_channel // 4, out_channel, 7, padding=0)

    def forward(self, x):
        out = F.relu(self.bn1(self.tconv1(x)), True)
        out = F.relu(self.bn2(self.tconv2(out)), True)
        out = self.rp(out)
        out = self.conv3(out)

        return F.tanh(out)


class Generator(nn.Module):
    def __init__(self, in_channel, sc_channel, out_channel, n_res):
        super(Generator, self).__init__()
        self.n_res = n_res
        self.downsample = DownSample(in_channel, sc_channel)
        self.resblock = ResBlock(sc_channel * 4)
        self.upsample = Upsample(sc_channel * 4, out_channel)

    def forward(self, x):
        out = self.downsample(x)
        for i in range(self.n_res):
            out = self.resblock(out)
        out = self.upsample(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel, sc_channel):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, sc_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(sc_channel, sc_channel * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(sc_channel * 2)
        self.conv3 = nn.Conv2d(sc_channel * 2, sc_channel * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(sc_channel * 4)
        self.conv4 = nn.Conv2d(sc_channel * 4, sc_channel * 8, 4, 1, 1)
        self.bn4 = nn.BatchNorm2d(sc_channel * 8)
        self.conv5 = nn.Conv2d(sc_channel * 8, 1, 4, 1, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2, True)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2, True)
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2, True)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2, True)
        out = self.conv5(out)

        return out
