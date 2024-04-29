"""
Created on Sat Jul 17 19:58:19 2021

@author: 逗号@东莞理工ACE实验室
"""

import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for ArmorNum dataset(3×48×48）
    """
    def __init__(self):
        # 调用类的初始化方法来初始化父类
        super(Lenet5, self).__init__()

        # 新建一个conv_unit变量
        # 用Sequential包含网络，可以方便地组织各种结构
        self.conv_unit = nn.Sequential(
            # 建立一个卷积层
            # x:[b, 3, 48, 48] => [b, 6, ?, ?] 大小size暂时未知，因为它与kernel_size、stride和padding有关，大概在32左右
            # 根据Yann LeCun的paper，第一个卷积层的输出是6个channels
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 在Yann LeCun的Lenet5 paper中，第二层是个Subsampling层，我们这里用pooling池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 第三层，第二个卷积层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # 第四层，再来一个池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            # 接下来是Full connection全连接层，需要用Flatten，将经过上面四层处理的输出原本是四维的转换成二维，因为Pytorch没有这个转换的类，所以不能在Sequential里完成Full connection,只能用view方法进行flatten
            # 因此建立两个unit，一个是这里的conv_unit，一个是下面的fc_unit
        )

        # Full connection unit，全连接层
        self.fc_unit = nn.Sequential(
            # 因为不知道conv_unit输出的shape是怎样的，因此下面nn.Linear的第一个参数需要通过下面的tmp和out测试conv_unit输出的shape来决定
            # 这里的16*5*5是已经通过测试得到的
            nn.Linear(16 * 9 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6)
        )

        # # 临时构造一个tmp数据，用于测试conv_unit到fc_unit的第一个Linear层的第一个参数是多少
        # # [b, 3, 48, 48]
        # tmp = torch.randn(2, 3, 48, 48)
        # out = self.conv_unit(tmp)
        # # [b, 16, 9, 9]
        # print('conv out:', out.shape)

        # # 因为softmax()函数的输出不稳定，因此这里使用包含了softmax()的CrossEntropyLoss
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: [b, 3, 48, 48]
        :return: logits
        """
        batchsz = x.size(0)
        # [b, 3, 48, 48] => [b, 16, 9, 9]
        x = self.conv_unit(x)
        # [b, 16, 9, 9] => [b, 16*9*9]
        x = x.view(batchsz, 16*9*9)     # 也可写成x = x.view(batchsz, -1)
        # [b, 16*9*9] => [b, 6]
        # 因为经过fc_unit全连接层之后，还要经过softmax()函数处理，这个在全连接层后softmax()前的输出就叫logits
        logits = self.fc_unit(x)

        return logits
