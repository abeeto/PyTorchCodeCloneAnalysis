#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:07:11 2018

@author: jangqh
"""
from __future__ import division

###
print """
两种卷积形式：
（一）  torch.nn.Conv2d(batch, channel， H, W)
（二）  troch.nn.functional.conv2d(batch, channel, H, W)
batch:输入的一批数据的数目
channnel：输入的通道数   彩色图片是3    灰色图片是1  
H：高
W：宽
（32， 3， 50， 100）表示一个batch是32张彩色图片，通道为3， 高宽分别为50， 100 
---------------------------------------------------

"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

###读入一张灰度图片
im = Image.open('./cat.png').convert('L')
im1 = plt.imread('./dog.png')


###转换为一个矩阵
im = np.array(im, dtype = 'float32')
print im.shape[0]
im1 = np.array(im1, dtype = 'float32')

###可视化
#plt.imshow(im.astype('uint8'), cmap = 'gray')
#plt.imshow(im)
plt.show()



print """
将图片转换成pytorch tensor格式  并适用于卷积输入要求
"""
print "numpy格式 reshape 之前的大小:", im.shape
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
print "torch tensor 格式 reshape 之后的大小:", im.size()


##使用nn.Conv2d

conv1 = nn.Conv2d(1, 1, 3, bias = False)   ###定义卷积
print conv1

###下面我们定义一个算子对其进行轮廓检测
sobel_kernel = np.array([[-1, -1, -1], [-1,8,-1], [-1, -1, -1]],dtype = 'float32')    ###定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1,1,3,3))    ###适配卷积的输入输出
conv1.weight.data = torch.from_numpy(sobel_kernel)   ####给卷积的kernel赋值

edge1 = conv1(Variable(im))  ##作用在图片
edge1 = edge1.data.squeeze().numpy()        ###将输出转换为图片的格式

                          
print "可视化边缘检测输出结果...."
plt.imshow(edge1, cmap = 'gray')
#plt.show()


print """
卷积网络中另外一个非常中重要的结构就是池化，这是利用了图片的下采样不变性，即一张图片变小了
还是能够看出了这张图片的内容，而使用池化层能够将图片的大小降低，非常好的提高了计算效率，
同时，池化层没有参数，有最大池化层，均值池化层，
（一）nn.MaxPool2d()
(二) torch.nnFunctional.max_pool2d()

"""

####
pool1 = nn.MaxPool2d(2,2)
print "before max pool,image shape:{} x {}".format(im.size()[2], im.size()[3])
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print "after max pool,image shape:{} x {}".format(small_im1.shape[0], small_im1.shape[1])

print "可视化 图片   池化之后.."

plt.imshow(small_im1, cmap = 'gray')
#plt.show()

print "使用（二） F.max_pool2d"

print "before max pool, image shape:{} x {}".format(im.size()[2], im.size()[3])
small_im2 = F.max_pool2d(Variable(im), 2, 2)
small_im2 = small_im2.data.squeeze().numpy()
print "after max pool, image shape :{} x {}".format(small_im2.shape[0], small_im2.shape[1])

plt.imshow(small_im2, cmap = 'gray')
#plt.show()



"""
批标准化:使用批标准能够得到非常好的手链结果
数据预处理：中心化和标准化。
中心化---修正数据的中心位置，每个特征维度上减去对应的均值，最后得到0均值的特征。
标准化---在数据变成0均值后，为了得到不同特征维度有着相同的规律，可以除以标准差
         近似为一个标准正态分布，也可以一句最大值和最小值将其转化为-1～1之间
"""
import  sys
sys.path.append('..')

import torch

def simple_batch_norm_1d(x, gama, beta):
    eps = 1e-5
    x_mean = torch.mean(x)  ##保留维度进行 batchcast
    print "x_mean:",x_mean
    x_var = torch.mean((x - x_mean) ** 2)
    print "x_var", x_var
    x_hat = torch.sqrt(x_var)
    print x_hat
    return gama.view_as(x_mean) * x_hat + beta.view_as(x_mean)

print """
我们验证以下是否对于任意的输入，输出会被标准化
"""
x = torch.arange(0, 15).view(5, 3)
gamma = torch.ones(x.size()[1])
beta = torch.zeros(x.size()[1])
print 'before bn:',x
print "gama:", gamma
print "beta:", beta
y = simple_batch_norm_1d(x, gamma, beta)
print "after bn:", y






























