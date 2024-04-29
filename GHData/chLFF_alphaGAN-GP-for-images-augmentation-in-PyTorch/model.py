import torch.nn as nn


class Code_Discriminator(nn.Module):
    def __init__(self, code_size=256, num_units=1024):
        super(Code_Discriminator, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(code_size, num_units),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(num_units, 1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channel: int = 32):
        super(Discriminator, self).__init__()

        c = channel

        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 1*256*256 -> 32*128*128
        self.downsample2 = nn.Sequential(
            nn.Conv2d(c, c*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c*2),
            nn.LeakyReLU(0.2)
        )  # 64*64*64
        self.downsample3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c*4),
            nn.LeakyReLU(0.2)
        )  # 128*32*32
        self.downsample4 = nn.Sequential(
            nn.Conv2d(c*4, c*8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c*8),
            nn.LeakyReLU(0.2)
        )   # 256*16*16
        self.downsample5 = nn.Sequential(
            nn.Conv2d(c*8, c*16, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c*16),
            nn.LeakyReLU(0.2)
        )   # 512*8*8
        self.downsample6 = nn.Sequential(
            nn.Conv2d(c*16, c*32, 4, stride=2, padding=1),
            nn.InstanceNorm2d(c*32),
            nn.LeakyReLU(0.2)
            # 1024*4*4
        )
        self.downsample7 = nn.Sequential(
            nn.Conv2d(c*32, 1, 4, stride=1, padding=0),
            # 1*1*1
        )

    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.downsample5(x)
        x = self.downsample6(x)
        x = self.downsample7(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channel: int = 32, out_class: int = 256):
        super(Encoder, self).__init__()

        c = channel

        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2)
        )  # 1*256*256 -> 32*128*128
        self.downsample2 = nn.Sequential(
            nn.Conv2d(c, c*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(0.2)
        )  # 64*64*64
        self.downsample3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*4),
            nn.LeakyReLU(0.2)
        )  # 128*32*32
        self.downsample4 = nn.Sequential(
            nn.Conv2d(c*4, c*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*8),
            nn.LeakyReLU(0.2)
        )   # 256*16*16
        self.downsample5 = nn.Sequential(
            nn.Conv2d(c*8, c*16, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*16),
            nn.LeakyReLU(0.2)
        )   # 512*8*8
        self.downsample6 = nn.Sequential(
            nn.Conv2d(c*16, c*32, 4, stride=2, padding=1),
            nn.BatchNorm2d(c*32),
            nn.LeakyReLU(0.2)
            # 1024*4*4
        )
        self.downsample7 = nn.Sequential(
            nn.Conv2d(c*32, out_class, 4, stride=1, padding=0),
            # 256*1*1
        )

    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.downsample5(x)
        x = self.downsample6(x)
        x = self.downsample7(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_size: int = 256, channel: int = 32):
        super(Generator, self).__init__()
        c = channel
        self.z_size = z_size

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(z_size, c*32, 4, 1, 0, bias=False),
            # 256*1*1 -> 1024*4*4
            nn.BatchNorm2d(c*32),
            nn.ReLU()
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 1024*8*8
            nn.Conv2d(c*32, c*16, 3, 1, 1, bias=False),
            # 512*8*8
            nn.BatchNorm2d(c*16),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 512*16*16
            nn.Conv2d(c*16, c*8, 3, 1, 1, bias=False),
            # 256*16*16
            nn.BatchNorm2d(c*8),
            nn.ReLU()
        )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 256*32*32
            nn.Conv2d(c*8, c*4, 3, 1, 1, bias=False),
            # 128*32*32
            nn.BatchNorm2d(c*4),
            nn.ReLU()
        )
        self.upsample5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 128*64*64
            nn.Conv2d(c*4, c*2, 3, 1, 1, bias=False),
            # 64*64*64
            nn.BatchNorm2d(c*2),
            nn.ReLU()
        )
        self.upsample6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 64*128*128
            nn.Conv2d(c*2, c, 3, 1, 1, bias=False),
            # 32*128*128
            nn.BatchNorm2d(c),
            nn.ReLU()
        )
        self.upsample7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 32*128*128
            nn.Conv2d(c, 1, 3, 1, 1, bias=False),
            # 1*256*256
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.z_size, 1, 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        x = self.upsample6(x)
        x = self.upsample7(x)
        return x
