import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, padding=0, stride=1, bnorm=True, leaky=True):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding, bias=False if bnorm else True)
        self.bnorm = nn.BatchNorm2d(ch_out, eps=1e-3) if bnorm else None
        self.leaky = nn.LeakyReLU(0.1) if leaky else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.leaky(x)
        return x


class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=2, padding=1):
        super(DownSample, self).__init__()
        self.conv_bn_layer = ConvBNLayer(ch_in, ch_out, filter_size=filter_size, stride=stride, padding=padding)
        self.ch_out = ch_out

    def forward(self, x):
        x = self.conv_bn_layer(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, filter_size=1, stride=1, padding=0)
        self.conv2 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class LayerWarp(nn.Module):
    def __init__(self, ch_in, ch_out, count):
        super(LayerWarp, self).__init__()
        self.basicblock0 = BasicBlock(ch_in, ch_out)
        # self.res_out_list = nn.ModuleList()

        self.res_outs = nn.ModuleList(BasicBlock(ch_out * 2, ch_out) for i in range(count - 1))

    def forward(self, inputs):
        out = self.basicblock0(inputs)
        for i, res_out in enumerate(self.res_outs):
            out = res_out(out)
        # out = self.res_out_list(out)
        return out


# net = LayerWarp(32 * (2 ** (1 + 1)), 32 * (2 ** 1), 5)
# print(net)

DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(nn.Module):
    def __init__(self):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(ch_in=3, ch_out=32, filter_size=3, stride=1, padding=1)
        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2)
        # self.darknet53_conv_block_list = []
        # self.downsample_list = []

        self.darknet53_conv_block_lists = nn.ModuleList(
            LayerWarp(32 * (2 ** (i + 1)), 32 * (2 ** i), stage) for i, stage in enumerate(self.stages))
        # self.add_module('stage_%d' % (i), LayerWarp(32 * (2 ** (i + 1)), 32 * (2 ** i), 5))
        # self.darknet53_conv_block_list.append('stage_%d' % (i))

        self.downsample_lists = nn.ModuleList(
            DownSample(ch_in=32 * (2 ** (i + 1)), ch_out=32 * (2 ** (i + 2))) for i in range(len(self.stages) - 1))
        # self.add_module('stage_%d_downsample' % i,
        #                 DownSample(ch_in=32 * (2 ** (i + 1)), ch_out=32 * (2 ** (i + 2))))
        #
        # self.downsample_list.append('stage_%d_downsample' % i)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        # print(self.darknet53_conv_block_lists[0])
        # print(self.darknet53_conv_block_lists[1])
        for i, darknet53_conv_list in enumerate(self.darknet53_conv_block_lists):
            out = darknet53_conv_list(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_lists[i](out)
        return blocks[-1:-4:-1]


# with torch.no_grad():
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     net = DarkNet53_conv_body().to(device)
#     # net1=ConvBNLayer(3,32).to(device)
#     x = torch.randn(1, 3, 640, 640).to(device)
#     x = x.float()
#     y1, y2, y3 = net(x)
#     print(net)
#     print(y1.shape, y2.shape, y3.shape)

class YoloDetectionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(YoloDetectionBlock, self).__init__()
        assert ch_out % 2 == 0, 'channal {} cannot be divided by 2.'.format(ch_out)
        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, filter_size=1, stride=1, padding=0)
        self.conv1 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1)
        self.conv2 = ConvBNLayer(ch_in=ch_out * 2, ch_out=ch_out, filter_size=1, stride=1, padding=0)
        self.conv3 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1)
        self.route = ConvBNLayer(ch_in=ch_out * 2, ch_out=ch_out, filter_size=1, stride=1, padding=0)
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# NUM_ANCHORS = 3
# NUM_CLASS = 7
# num_filters = NUM_ANCHORS * (NUM_CLASS + 5)
# with torch.no_grad():
#     backbone = DarkNet53_conv_body().to(device)
#     detection = YoloDetectionBlock(ch_in=1024, ch_out=512).to(device)
#     conv2d_pred = torch.nn.Conv2d(1024, num_filters, 1)
#     x = torch.randn(1, 3, 640, 640).to(device)
    # x = x.float()
    # c0, c1, c2 = backbone(x)
    # print(c0.shape)
    # route, tip = detection(c0)
    # conv2d_pred = torch.nn.Conv2d(1024, num_filters, 1, padding=0).to(device)
    # print(route.shape)
    # print(tip.shape)
    # p0 = conv2d_pred(tip)
    # print(p0.shape)
