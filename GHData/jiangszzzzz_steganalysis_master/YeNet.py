import os
import sys
import numpy as np
from torchstat import stat
from thop import profile

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

SRM_npy = np.load('SRM_Kernels.npy')
# print(SRM_npy.shape)

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)

class SRM_RGB(nn.Module):
    def __init__(self):
        super(SRM_RGB, self).__init__()
        self.preprocessing = SRM_conv2d(1, 0)

    def forward(self, x):
        self.preprocessing.eval()
        ins0 = x[:, 0, :, :].unsqueeze(1)
        ins1 = x[:, 1, :, :].unsqueeze(1)
        ins2 = x[:, 2, :, :].unsqueeze(1)
        self.preprocessing(ins0)
        ins = torch.cat([self.preprocessing(ins0),
                        self.preprocessing(ins1),
                        self.preprocessing(ins2)], 1)
        return ins


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class YeNet(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(YeNet, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_RGB()
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        self.block2 = ConvBlock(90, 90, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(90, 90, 3, with_bn=self.with_bn)
        self.block4 = ConvBlock(90, 90, 3, with_bn=self.with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(90, 64, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(64, 64, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(64, 64, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(64, 32, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(32, 32, 3, 3, with_bn=self.with_bn)
        # 输出四分类
        self.ip1 = nn.Linear(8 * 8 * 32, 4)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.ip1(x)
        return x

    # 参数初始化
    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0., 0.01)
                mod.bias.data.zero_()

def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()



if __name__ == "__main__":
    x = torch.randn(size=(32, 3, 512, 512))  # 输入 1表示 batchsize 为一张图片， 3表示通道数， 256*256大小的图片
    print(f'输入的维度：{x.shape}')
    # print(x[:, 1, :, :])

    net = YeNet()

    # stat 查看模型的详细信息

    # stat(net, (3,224,224),)

    # flops, params = profile(net, inputs=(x,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    output_Y = net(x)
    print('output shape: ', output_Y.shape)
    # print('output: ', output_Y)
