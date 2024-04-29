import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)

        return x


# N_dims = sqrt[(H/8)x(W/8)]
class unFlatten(nn.Module):
    def __init__(self, n_dims):
        super(unFlatten, self).__init__()
        self.n_dims = n_dims

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], self.n_dims, self.n_dims)

        return x


class ConvBlock1(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=1, padding=0, dilation=1):
        super(ConvBlock1, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=2, padding=0, dilation=1):
        super(ConvBlock2, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class DeConvBlock1(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3, 3), stride=1, padding=0, dilation=1):
        super(DeConvBlock1, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class DeConvBlock2(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(4, 4), stride=2, padding=1, dilation=1):
        super(DeConvBlock2, self).__init__()

        conv_block = [
            nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(c_out),
            nn.PReLU()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


# EVD = Eigen Value Decomposition
class EVD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        # y = x.view(x.shape[0], x.shape[1], -1)
        S = torch.bmm(y, y.transpose(2, 1)) / y.shape[-1]
        D, U = torch.linalg.eig(S)
        U = U.real
        y_rot = torch.matmul(U, y)
        return y_rot, U


class InverseEVD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, U, y_wave):
        y_est = torch.matmul(U.transpose(2, 1), y_wave)
        ################## FOR DEBUG####################
        # B = torch.matmul(U.transpose(2, 1), U)
        # print(B)
        return y_est


# add Uniform noise for training only:
class addUniformNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_rot):
        mu = 1 / (2 ** 10)
        Noise = (2 * torch.rand_like(y_rot) - 1) * mu
        y_tag = y_rot + Noise
        return y_tag #y_rot.add_(Noise)


# Quantization for testing the algorithm
class QuantizationBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_rot):
        # B = the number of bits for the desired precision
        B = 12
        # need to add ROUND function, but wo using un-diferentiable functions inside
        y_tag = torch.round((2 ** (B - 1)) * y_rot)
        return y_tag, B


class deQuantizationBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, B, y_tag):
        # B = the number of bits for the desired precision
        y_wave = y_tag / (2 ** (B - 1))
        return y_wave


# c1 = num of input channels
# ci = num of filters/features in the output of the previous layer
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = ConvBlock1(c_in=1, c_out=32, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l2 = ConvBlock2(32, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.l3 = ConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l4 = ConvBlock2(64, 64, kernel_size=3, stride=2, padding=0, dilation=1)
        self.l5 = ConvBlock1(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l6 = ConvBlock2(64, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        ################## FOR DEBUG####################
        # self.l6 = ConvBlock2(64, 2, kernel_size=3, stride=2, padding=0, dilation=1)

        self.flat = Flatten()  # Avi's function
        # self.fc1 = nn.Linear(32, latent_dim)  # Not sure it's necessary

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = self.flat(x)

        # x = self.fc1(self.flat(x))
        return x


class Decoder(nn.Module):
    def __init__(self, image_size):
        super(Decoder, self).__init__()
        self.l1 = DeConvBlock1(c_in=32, c_out=32, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l2 = DeConvBlock2(32, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        self.l3 = DeConvBlock1(32, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l4 = DeConvBlock2(64, 64, kernel_size=4, stride=2, padding=1, dilation=1)
        self.l5 = DeConvBlock1(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.l6 = DeConvBlock2(64, 32, kernel_size=4, stride=2, padding=1, dilation=1)
        # fl = Final Layer, returns x to be Image
        self.fl = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, dilation=1),        
            nn.ReLU(inplace=True),
        )

        self.unflat = unFlatten(image_size // 8)  # Avi's function

    def forward(self, x):
        x = self.unflat(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = self.fl(x)

        # x = self.fc1(self.flat(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, image_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.rotation = EVD()
        self.quantization = addUniformNoise()
        self.inverserotation = InverseEVD()
        self.decoder = Decoder(image_size)

    def forward(self, x):
        y = self.encoder(x)
        # print(y)
        # y_rot = self.rotation(y)
        # print(y_rot)
        # y_tag = self.quantization(y_rot)
        # print(y_tag)
        # y_est = self.inverserotation(U, y_tag)
        # print(y_est)
        # x_est = self.decoder(y_est)
        x_est = self.decoder(y)
        # print(x_est)
        return y, x_est


# class AutoEncoderTest(nn.Module):
#     def __init__(self, image_size):
#         super(AutoEncoder, self).__init__()
#         self.encoder = Encoder()
#         self.rotation = EVD()
#         self.quantization = QuantizationBlock()
#         # self.entropycoding = EntropyCoder()
#         # self.entropydecoding = EntropyDeCoder()
#         self.dequantization = deQuantizationBlock()
#         self.inverserotation = InverseEVD()
#         self.decoder = Decoder(image_size)
#
#     def forward(self, x):
#         y = self.encoder(x)
#         # print(y)
#         y_rot, U = self.rotation(y)
#         # print(y_rot)
#         y_tag_in = self.quantization(y_rot)
#         # print(y_tag)
#         # y_code = entropycoding(y_tag_in)
#         # y_tag_out = entropydecoding(y_code)  # maybe just take y_tag_in
#         y_wave = self.dequantization(y_tag_out)  # maybe just take y_tag_in
#         y_est = self.inverserotation(U, y_wave)  # maybe just take y_tag_in
#         # print(y_est)
#         # x_est = self.decoder(y_est)
#         x_est = self.decoder(y)
#         # print(x_est)
#         return y_code, x_est


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128)
    # E = Encoder()
    # y = E(x)
    # print(y.shape)
    #
    # D = Decoder()
    # z = D(y)
    # print(z.shape)

    A = AutoEncoder(128)
    f1, f2 = A(x)
    print(f1.shape, f2.shape)
