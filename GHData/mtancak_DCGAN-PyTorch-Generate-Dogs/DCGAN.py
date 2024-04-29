import torch
import torch.nn as nn
from enum import Enum


class GANMode(Enum):
    DISCRIMINATOR = 1
    GENERATOR = 2


class DCGAN(nn.Module):
    def __init__(self, shape, in_channels, out_channels, mode_limit):
        super(DCGAN, self).__init__()

        # initialise values
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = GANMode.DISCRIMINATOR
        self.mode_executions = 0
        self.mode_limit = mode_limit

        # create discriminator and generator
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4),
            nn.Softmax(dim=1)
        )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.mode == GANMode.DISCRIMINATOR:
            return self.discriminator(x)
        elif self.mode == GANMode.GENERATOR:
            return self.generator(x)
        else:
            print("ERROR")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = DCGAN((64, 64), 3, 1, 10).to(DEVICE)
    x = torch.rand(1, 100, 1, 1).to(DEVICE)

    # test generator
    model.mode = GANMode.GENERATOR
    x_ = model(x)
    print(x_.shape)

    # test discriminator
    model.mode = GANMode.DISCRIMINATOR
    x__ = model(x_)
    print(x__.shape)
