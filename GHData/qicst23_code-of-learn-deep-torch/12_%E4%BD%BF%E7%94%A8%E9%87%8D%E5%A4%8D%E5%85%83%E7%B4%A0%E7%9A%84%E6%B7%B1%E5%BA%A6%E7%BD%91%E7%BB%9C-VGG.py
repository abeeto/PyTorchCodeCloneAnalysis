#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 23:37:48 2018

@author: jangqh
"""

print """
CIFAR10 : 50000张训练集，10000张测试集，图片大小32x32x3，一共十类问题

"""

print"""
VGGNET：不断的堆叠卷积层和池化层，几乎全部使用3x3的卷积核和2x2的池化层，使用小的卷积核进行多层
堆叠和一个大的卷积核的感受野是相同的，同时减少卷积核能减少参数，同时也可以有更深的结构。 
"""

import sys
sys.path.append('..')

import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

print """
我们可以定义一个vgg 的block,传入三个参数，
1--模型层数
2--输入通道数
3--输出通道数
第一层卷积层接受的输入通道数就是图片输入的通道数，然后输出的输出通道数，后面的卷积核接受的
通道数就是最后输出的通道数。
"""
def vgg_block(num_convs, in_channels, out_channels):
    ##定义第一层
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]
    
    #定义后面的多层
    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    
    ##定义池化层
    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)

print "打印出来模型的结构："
block_demo = vgg_block(3, 64, 128)
print (block_demo)

###首先定义输入为：[1, 64, 300, 300]
input_demo = Variable(torch.zeros(1, 64, 300, 300))
output_demo = block_demo(input_demo)

print(output_demo.size())
























