import torch
import torch.nn as nn
from config import args


class ResidualConv(nn.Module):
    def __init__(self, chan, chan2):
        super(ResidualConv, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(chan, chan2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(chan2, chan2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_up = nn.Conv2d(chan, chan2, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(chan2, momentum=args.momentum)
        self.no_change_conv = nn.Conv2d(chan, chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.no_change_bn = nn.BatchNorm2d(chan, momentum=args.momentum)

    def forward(self, img, change):
        if change:
            x = self.lrelu(self.bn(self.conv1(img)))
            x = self.bn(self.conv2(x))
            img = self.bn(self.conv_up(img))
            out = self.lrelu(img + x)
            return out
        else:
            x = self.lrelu((self.no_change_bn(self.no_change_conv(img))))
            x = self.no_change_bn(self.no_change_conv(x))
            out = self.lrelu(img + x)
            return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(args.n_channels, int(args.h_dim/16), kernel_size=3, stride=1, padding=1, bias=False)# (64, 64)
        self.conv1_bn = nn.BatchNorm2d(int(args.h_dim/16), momentum=args.momentum)
        self.conv4 = nn.Conv2d(args.h_dim, args.latent_dim, int(args.img_size/16), 1, 0, bias=False)# (batch_size, latent_h_dim, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(args.latent_dim, momentum=args.momentum)
        #self.linear = nn.Linear(args.latent_dim, args.latent_dim, bias=False)
        self.residual_conv1 = ResidualConv(int(args.h_dim/16), int(args.h_dim/8))
        self.residual_conv2 = ResidualConv(int(args.h_dim/8), int(args.h_dim/4))
        self.residual_conv3 = ResidualConv(int(args.h_dim/4), int(args.h_dim/2))
        self.residual_conv4 = ResidualConv(int(args.h_dim/2), args.h_dim)
        self.residual_conv5 = ResidualConv(args.h_dim, args.h_dim)

        if torch.cuda.is_available():
            self.residual_conv1.cuda()
            self.residual_conv2.cuda()
            self.residual_conv3.cuda()
            self.residual_conv4.cuda()
            self.residual_conv5.cuda()

    def forward(self, img):
        #print(img.shape)
        x = self.conv1_bn(self.lrelu(self.conv1(img)))
        #print(x.shape)
        x = self.residual_conv1(x, change=False)
        x = self.residual_conv1(x, change=False)
        x = self.residual_conv1(x, change=True)
        #print(x.shape)
        x = self.residual_conv2(x, change=False)
        x = self.residual_conv2(x, change=True)
        #print(x.shape)
        x = self.residual_conv3(x, change=False)
        x = self.residual_conv3(x, change=True)
        #print(x.shape)
        x = self.residual_conv4(x, change=False)
        x = self.residual_conv4(x, change=True)
        #print(x.shape)
        x = self.residual_conv5(x, change=False)
        #x = self.conv4_bn(self.lrelu(self.conv4(x)))
        x = self.conv4(x)
        #print(x.shape)
        z = x.view(x.shape[0], -1)
        #z = self.linear(x)
       # logvar = self.linear(x)
       # z = reparameterization(mu, logvar)
        return z


class ResidualDeconv(nn.Module):
    def __init__(self, chan, chan2):
        super(ResidualDeconv, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(chan, chan2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(chan2, chan2, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv_down = nn.ConvTranspose2d(chan, chan2, kernel_size=1, stride=2, padding=0, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(chan2, momentum=args.momentum)
        self.no_change_deconv = nn.ConvTranspose2d(chan, chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.no_change_bn = nn.BatchNorm2d(chan, momentum=args.momentum)

    def forward(self, img, change):
        if change:
            x = self.lrelu(self.bn(self.deconv1(img)))
            x = self.bn(self.deconv2(x))
            img = self.bn(self.deconv_down(img))
            out = self.lrelu(img + x)
            return out
        else:
            x = self.lrelu((self.no_change_bn(self.no_change_deconv(img))))
            x = self.no_change_bn(self.no_change_deconv(x))
            out = self.lrelu(img + x)
            return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(args.latent_dim, args.h_dim, int(args.img_size/16), 1, 0, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(args.h_dim, momentum=args.momentum)
        self.deconv4 = nn.ConvTranspose2d(int(args.h_dim/16), args.n_channels, 3, 1, 1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(args.n_channels, momentum=args.momentum)
        self.tanh = nn.Tanh()

        self.residual_deconv1 = ResidualDeconv(args.h_dim, int(args.h_dim/2))
        self.residual_deconv2 = ResidualDeconv(int(args.h_dim/2), int(args.h_dim/4))
        self.residual_deconv3 = ResidualDeconv(int(args.h_dim/4), int(args.h_dim/8))
        self.residual_deconv4 = ResidualDeconv(int(args.h_dim/8), int(args.h_dim/16))
        self.residual_deconv5 = ResidualDeconv(int(args.h_dim/16), int(args.h_dim/16))

        if torch.cuda.is_available():
            self.residual_deconv1.cuda()
            self.residual_deconv2.cuda()
            self.residual_deconv3.cuda()
            self.residual_deconv4.cuda()
            self.residual_deconv5.cuda()

    def forward(self, z):
        #z = z.view(z.shape[0], args.latent_dim, 1, 1)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        z = self.lrelu(self.deconv1_bn(self.deconv1(z)))
        #print(z.shape)
        x = self.residual_deconv1(z, change=False)
        x = self.residual_deconv1(x, change=False)
        x = self.residual_deconv1(x, change=True)
        #print(x.shape)
        x = self.residual_deconv2(x, change=False)
        x = self.residual_deconv2(x, change=True)
        #print(x.shape)
        x = self.residual_deconv3(x, change=False)
        x = self.residual_deconv3(x, change=True)
        #print(x.shape)
        x = self.residual_deconv4(x, change=False)
        x = self.residual_deconv4(x, change=True)
        #print(x.shape)
        x = self.residual_deconv5(x, change=False)

        out1 = self.deconv4_bn(self.deconv4(x))
        #print(out1.shape)
        #img1 = self.sigmoid(out1)
        img1 = self.tanh(out1)
        return img1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, int(args.latent_dim/2), bias=False),
            nn.BatchNorm1d(int(args.latent_dim/2), momentum=args.momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(args.latent_dim/2), int(args.latent_dim/4), bias=False),
            nn.BatchNorm1d(int(args.latent_dim/4), momentum=args.momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(args.latent_dim/4), 1, bias=False)
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
