# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 16:41
# @Author  : zwenc
# @File    : NetNN.py

from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # 卷积层，输入一张图片，输出6张，滤波器为5*5大小，cuda表示使用GPU计算
        self.conv1 = nn.Conv2d(1,6,5).cuda()
        self.conv2 = nn.Conv2d(6,16,5).cuda()
        self.fc1 = nn.Linear(16 * 4 * 4,120).cuda()
        self.fc2 = nn.Linear(120,84).cuda()
        self.fc3 = nn.Linear(84,10).cuda()

    # 继承来自nn.Module的接口，必须实现，不能改名。
    # max_pool2d，池化函数，用来把图像缩小一半
    # relu 神经元激励函数，y = max(x,0)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))

        x = x.view(-1,self.numOfWidth_X_Height_X_num(x)) # 类似于reshape功能，重塑张量形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def numOfWidth_X_Height_X_num(self,x):
        """
        :param x: 输入张量，有多个组，每个组有很多个单位，每个单位有1~3个张量。灰度图为1个张量，彩色图为3个
        :return: 每个组有多少个像素
        """
        # 第0位表示有多少个组，所以从1开始。
        # torch.Size([60, 16, 4, 4]), 有60个组，每个组16个张量，张量大小为4*4。
        size = x.size()[1:]
        temp = 1
        for s in size:
            temp *= s

        return temp