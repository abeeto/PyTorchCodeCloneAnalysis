#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2022/2/6 11:51
@File:          efficinetnetv2.py
'''

from collections import OrderedDict
import copy
import torch
import math
from torch import nn
from torch.nn import functional as F

def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SE(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SE, self).__init__()
        inter_channels = max(1, int(channels * se_ratio))
        self.conv1 = nn.Conv2d(channels, inter_channels, (1, 1))
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.silu(self.conv1(input))
        x = self.sigmoid(self.conv2(x))
        return x * input

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.0,
                 survival_probability=0.2):
        super(MBConvBlock, self).__init__()
        self.stage1 = nn.Sequential()
        channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.stage1.add_module('conv1', nn.Conv2d(in_channels, channels, (1, 1), bias=False))
            self.stage1.add_module('bn1', nn.BatchNorm2d(channels))
            self.stage1.add_module('silu1', nn.SiLU(inplace=True))

        self.stage2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(channels, channels, (kernel_size, kernel_size), stride=(stride, stride),
                                padding=(kernel_size // 2, kernel_size // 2), bias=False, groups=channels)),
            ('bn2', nn.BatchNorm2d(channels)),
            ('silu2', nn.SiLU(inplace=True))
        ]))
        if 0 < se_ratio <= 1:
            self.se = SE(channels, se_ratio)
        else:
            self.se = nn.Sequential()
        self.stage3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(channels, out_channels, (1, 1), bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels))
        ]))

        self.use_shortcut = (in_channels == out_channels and stride == 1)
        self.survival_probability = survival_probability

    def forward(self, input):
        x = self.stage1(input)
        x = self.stage2(x)
        x = self.se(x)
        x = self.stage3(x)
        if self.use_shortcut:
            if 0 < self.survival_probability < 1:
                x = F.dropout(x, p=self.survival_probability, training=self.training, inplace=True)
            x = x + input

        return x

class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.0,
                 survival_probability=0.2):
        super(FusedMBConvBlock, self).__init__()
        self.stage1 = nn.Sequential()
        channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.stage1.add_module('conv1', nn.Conv2d(in_channels, channels, (kernel_size, kernel_size),
                                                      stride=(stride, stride),
                                                      padding=(kernel_size // 2, kernel_size // 2), bias=False))
            self.stage1.add_module('bn1', nn.BatchNorm2d(channels))
            self.stage1.add_module('silu1', nn.SiLU(inplace=True))

        if 0 < se_ratio <= 1:
            self.se = SE(channels, se_ratio)
        else:
            self.se = nn.Sequential()
        self.stage2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(channels, out_channels, (1, 1), bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))
        if expand_ratio == 1:
            self.stage2.add_module('silu2', nn.SiLU(inplace=True))

        self.use_shortcut = (in_channels == out_channels and stride == 1)
        self.survival_probability = survival_probability

    def forward(self, input):
        x = self.stage1(input)
        x = self.se(x)
        x = self.stage2(x)
        if self.use_shortcut:
            if 0 < self.survival_probability < 1:
                x = F.dropout(x, p=self.survival_probability, training=self.training, inplace=True)
            x = x + input

        return x

class EfficientNetV2(nn.Module):
    def __init__(self, blocks_args, width_coefficient, depth_coefficient, dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=10000):
        super(EfficientNetV2, self).__init__()
        # Build stem
        stem_channels = _make_divisible(blocks_args[0]['in_channels'] * width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )

        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)
        b = 0
        num_blocks = float(sum(args['num_repeat'] for args in blocks_args))
        layers = []
        for (i, args) in enumerate(blocks_args):
            assert args['num_repeat'] > 0

            # Update block input and output filters based on depth multiplier.
            args['in_channels'] = _make_divisible(args['in_channels'] * width_coefficient)
            args['out_channels'] = _make_divisible(args['out_channels'] * width_coefficient)

            # Determine which conv type to use:
            block = {0: MBConvBlock, 1: FusedMBConvBlock}[args['conv_type']]
            num_repeats = int(math.ceil(args['num_repeat'] * depth_coefficient))

            for j in range(num_repeats):
                # The first block needs to take care of stride and filter size increase.
                if j > 0:
                    args['stride'] = 1
                    args['in_channels'] = args['out_channels']

                layers.append(block(args['in_channels'], args['out_channels'], expand_ratio=args['expand_ratio'],
                                    kernel_size=args['kernel_size'], stride=args['stride'], se_ratio=args['se_ratio'],
                                    survival_probability=drop_connect_rate * b / num_blocks))
        self.blocks = nn.Sequential(*layers)

        # Build top
        top_channels = _make_divisible(1280 * width_coefficient)
        self.top = nn.Sequential(
            nn.Conv2d(blocks_args[-1]['out_channels'], top_channels, (1, 1), bias=False),
            nn.BatchNorm2d(top_channels),
            nn.SiLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if 0 < dropout_rate < 1:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        else:
            self.dropout = nn.Sequential()
        self.classifier = nn.Linear(top_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.top(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

def efficientnetv2_b0(**kwargs):
    blocks_args = [{
        'kernel_size': 3,
        'num_repeat': 1,
        'in_channels': 32,
        'out_channels': 16,
        'expand_ratio': 1,
        'se_ratio': 0,
        'stride': 1,
        'conv_type': 1
    }, {
        'kernel_size': 3,
        'num_repeat': 2,
        'in_channels': 16,
        'out_channels': 32,
        'expand_ratio': 4,
        'se_ratio': 0,
        'stride': 2,
        'conv_type': 1
    }, {
        'kernel_size': 3,
        'num_repeat': 2,
        'in_channels': 32,
        'out_channels': 48,
        'expand_ratio': 4,
        'se_ratio': 0,
        'stride': 2,
        'conv_type': 1
    }, {
        'kernel_size': 3,
        'num_repeat': 3,
        'in_channels': 48,
        'out_channels': 96,
        'expand_ratio': 4,
        'se_ratio': 0.25,
        'stride': 2,
        'conv_type': 0
    }, {
        'kernel_size': 3,
        'num_repeat': 5,
        'in_channels': 96,
        'out_channels': 112,
        'expand_ratio': 6,
        'se_ratio': 0.25,
        'stride': 1,
        'conv_type': 0
    }, {
        'kernel_size': 3,
        'num_repeat': 8,
        'in_channels': 112,
        'out_channels': 192,
        'expand_ratio': 6,
        'se_ratio': 0.25,
        'stride': 2,
        'conv_type': 0
    }]
    return EfficientNetV2(blocks_args, 1.0, 1.0, **kwargs)