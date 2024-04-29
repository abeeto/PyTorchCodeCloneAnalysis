# -*- coding: UTF-8-*-
import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.optim as optim

"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/23 10:21
"""

# 生成500个随机样本，每个样本20个特征
X = torch.rand((500, 20))
# 三分类问题，我们吧0、1、2代指分类结果。 结果为500行1列。
y = torch.randint(low=0, high=3, size=(500,), dtype=torch.float32)

input_ = X.shape[1]
output_ = len(y.unique())
# 步长 learning rate
lr = 0.07

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
# zhat = net.forward(X)
# loss = criterion(zhat, y.long())
# # 反向传播
# loss.backward()  # retain_graph = True 重复计算

# # 权重 w(t)
# w = net.linear1.weight.data
# # 梯度 d(w)
# dw = net.linear1.weight.grad
# # w_(t + 1) = w_t - η * (∂L / ∂w_t)
# w = w - lr * dw

# 动量法，让起始点了解过去的方向来确定移动的距离大小
"""
方法：让上一步的梯度向量的反方向，与现在这一点的梯度向量的反方向，进行加权求和。
动量v
    v(t) = γ* v(t-1) - η * L / ∂W
    w(t+1) = w(t) + v(t)
"""
gamma = 0.9
# w = net.linear1.weight.data
# print(w.shape)
# dw = net.linear1.weight.grad
# print(dw.shape)
# # 首次迭代的时候需要定义一个v0
# # 矩阵相减同型，那么即v初始矩阵和w权重矩阵同型，
# v = torch.zeros(w.shape[0], w.shape[1])
#
# v = gamma * v - lr * dw
# w = w + v

# torch官方实现 torch.optim
# 小批量随机梯度下降SGD
opt = optim.SGD(
    # 需要进行迭代的权重，前面笔记里说明了parameters包含所有计算结果
    net.parameters(),
    lr=lr,
    momentum=gamma
)

"""
梯度下降流程:
1.向前传播
2.计算本轮损失函数
3.反向传播 - 得到了梯度
4.更新权重和动量
5.清空梯度 - 清除原来计算出来的，基于上一个点的坐标计算的梯度（不清楚，即保留）
6.1 2 3 4 5迭代运行修正权重和动量。
"""
zhat = net.forward(X)
loss = criterion(zhat, y.long())
loss.backward()
opt.step()  # 移动一步，更新w和v
opt.zero_grad()

print(loss)
print(net.linear1.weight.data)
