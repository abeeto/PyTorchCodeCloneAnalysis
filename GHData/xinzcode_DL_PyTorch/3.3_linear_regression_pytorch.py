import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_pytorch.utils import *
import torch.utils.data as Data
from torch.nn import init  # PyTorch在init模块中提供了多种参数初始化方法。
import torch.nn
import torch.optim as optim  # torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size = 10
num_epochs = 3

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 每次都是随机打乱，然后分成大小为n的若干个mini-batch.
# 读取并打印第一个小批量数据样本
# for X, y in data_iter:
#     print(X, y)
#     break

# net = LinearNet(num_inputs)
# print(net) # 使用print可以打印出网络的结构

# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1),  # in_features=2, out_features=1 bias=true, 即考虑偏置的情况
    # 此处还可以传入其他层
    )

# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
#
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))
# print(net)
# print(net[0])

# 初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)  # 将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)  偏差会初始化为零。

# 损失函数
loss = nn.MSELoss()  # 均方误差损失

# 优化算法  定义一个优化器实例
optimizer = optim.SGD(net.parameters(), lr=0.03)  # 指定学习率为0.03的小批量随机梯度下降（SGD）
# print(optimizer)

# 为不同子网络设置不同的学习率，这在finetune时经常用到。
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

# —新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
# 但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

# 训练模型
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 比较学到的模型参数和真实的模型参数
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)