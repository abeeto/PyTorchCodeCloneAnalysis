import torch.nn as nn
from config import args


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.deconv1 = nn.ConvTranspose2d(args.latent_dim, args.h_dim, kernel_size=4, stride=1, padding=0, output_padding=0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(args.h_dim, int(args.h_dim/2), kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.deconv3 = nn.ConvTranspose2d(int(args.h_dim/2), int(args.h_dim/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.deconv4 = nn.ConvTranspose2d(int(args.h_dim/4), int(args.h_dim/8), kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.deconv5 = nn.ConvTranspose2d(int(args.h_dim/8), args.n_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(args.h_dim, momentum=args.momentum)
        self.bn2 = nn.BatchNorm2d(int(args.h_dim/2), momentum=args.momentum)
        self.bn3 = nn.BatchNorm2d(int(args.h_dim/4), momentum=args.momentum)
        self.bn4 = nn.BatchNorm2d(int(args.h_dim/8), momentum=args.momentum)

    def forward(self, img):
        x1 = self.lrelu(self.bn1(self.deconv1(img)))
        x2 = self.lrelu(self.bn2(self.deconv2(x1)))
        x3 = self.lrelu(self.bn3(self.deconv3(x2)))
        x4 = self.lrelu(self.bn4(self.deconv4(x3)))
        x5 = self.tanh(self.deconv5(x4))
        return x5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(args.n_channels, int(args.h_dim/8), kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(int(args.h_dim/8), int(args.h_dim/4), kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(int(args.h_dim/4), int(args.h_dim/2), kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(int(args.h_dim/2), args.h_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(args.h_dim, args.latent_dim, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.h_dim/8), momentum=args.momentum)
        self.bn2 = nn.BatchNorm2d(int(args.h_dim/4), momentum=args.momentum)
        self.bn3 = nn.BatchNorm2d(int(args.h_dim/2), momentum=args.momentum)
        self.bn4 = nn.BatchNorm2d(args.h_dim, momentum=args.momentum)

    def forward(self, img):
        x1 = self.lrelu(self.bn1(self.conv1(img)))
        x2 = self.lrelu(self.bn2(self.conv2(x1)))
        x3 = self.lrelu(self.bn3(self.conv3(x2)))
        x4 = self.lrelu(self.bn4(self.conv4(x3)))
        x5 = self.tanh(self.conv5(x4))
        return x5
