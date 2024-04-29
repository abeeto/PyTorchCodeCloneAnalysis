# -*- coding: UTF-8-*-
import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/23 10:21
梯度下降流程:
1.向前传播
2.计算本轮损失函数
3.反向传播 - 得到了梯度
4.更新权重和动量
5.清空梯度 - 清除原来计算出来的，基于上一个点的坐标计算的梯度（不清楚，即保留）
6.1 2 3 4 5迭代运行修正权重和动量。
"""

# 生成500个随机样本，每个样本20个特征
X = torch.rand((500, 20))
# 三分类问题，我们吧0、1、2代指分类结果。 结果为500行1列。
y = torch.randint(low=0, high=3, size=(500,), dtype=torch.float32)

input_ = X.shape[1]
output_ = len(y.unique())
# 步长 learning rate
lr = 0.07
gamma = 0.9

torch.manual_seed(996)


class Model(nn.Module):
    """
    参见L12
    """

    def __init__(self, in_features=10, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 13, bias=False)
        self.linear2 = nn.Linear(13, 8, bias=False)
        self.output = nn.Linear(8, out_features, bias=True)

    def forward(self, x):
        sigma1 = torch.relu(self.linear1(x))
        sigma2 = torch.sigmoid(self.linear2(sigma1))
        zhat = self.output(sigma2)
        return zhat


net = Model(in_features=input_, out_features=output_)
criterion = nn.CrossEntropyLoss()

# torch官方实现 torch.optim
# 小批量随机梯度下降SGD
opt = optim.SGD(
    # 需要进行迭代的权重，前面笔记里说明了parameters包含所有计算结果
    net.parameters(),
    lr=lr,
    momentum=gamma
)

# 如果对这部分进行迭代，如果数据集巨大，则必然效率极低
# for i in ...:
#     zhat = net.forward(X)
#     loss = criterion(zhat, y.long())
#     loss.backward()
#     opt.step()
#     opt.zero_grad()

"""
mini-batch SGD 梯度下降实际是抽样数据集进行的。有利有弊。
优点：有可能跳过局部最小值，从而得到尽可能的全局最优。
缺点：迭代次数不稳定。

batch_size：批量的尺寸，样本量。
epochs：定义全体数据一共被学习了多少次。

设batch_size = N_B, 数据集总共m个样本。
epoch = n = m / N_B
"""

a = torch.randn(500, 2, 3)  # 500个2行3列
b = torch.randn(500, 3, 4, 5)  # 500个3维的tensor中有二维的tensor四行五列
c = torch.randn(500, 1)

# TensorDataset 用来合并和打包数据集，被合并对象第一维度上的值相等,否则报错。
dataset = TensorDataset(a, b, c)

batch_size = 120
# DataLoader 用来划分数据集
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    # shuffle 是否要随机打乱
    shuffle=True,
    # 120，120，120，120，20
    # 是否舍弃最后一个batch，因为最后一个可能不完整。
    drop_last=True
)
# 现在一共有多少哥哥batch
print(len(data_loader))
# 现在一共有多少组样本
print(len(data_loader.dataset))
