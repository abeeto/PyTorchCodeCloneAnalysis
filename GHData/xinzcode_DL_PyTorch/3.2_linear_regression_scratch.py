import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_pytorch.utils import *

num_inputs = 2       # 特征数
num_examples = 1000  # 训练数据集样本数
true_w = [2, -3.4]   # 权重
true_b = 4.2         # 偏差
batch_size = 10      # 批量大小
lr = 0.03            # 学习率
num_epochs = 5       # 迭代次数
net = linreg         # 线性回归的矢量计算表达式
loss = squared_loss  # 平方损失

# 生成数据集
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 创建1000行2列标准正态分布Tensor
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 生成标签 y=Xw+b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# 标签加上噪声项 y=Xw+b+ϵ 真实数据，用来计算损失
# 噪声项 ϵ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。
# features的每一行是一个长度为2的向量，而labels的每一行是一个长度为1的向量（标量）
# print(features[0], labels[0])

# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# # 第二个特征features[:, 1]和标签 labels 的散点图
# plt.show()

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break  # 输出第一批

# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零，每个小批量样本清一次
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# 比较学到的参数和用来生成训练集的真实参数。它们应该很接近。
print(true_w, '\n', w)
print(true_b, '\n', b)
