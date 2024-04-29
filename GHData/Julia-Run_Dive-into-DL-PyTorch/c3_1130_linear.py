import torch
import numpy as np
from matplotlib import pyplot as plt

# 数据集生成
true_w = [3.2, -8]
true_b = 10
num_examples = 1000
num_inputs = len(true_w)
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)  # dtype必须要统一
labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, labels.size()), dtype=torch.float)  # dtype必须要统一

# data_ iter
import torch.utils.data as Data  # import——01 torch.utils.data

batch_size = 10
dataset = Data.TensorDataset(features, labels)  # TensorDataset 和 Dataset
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # shuffle拼写
for x, y in data_iter:
    print(x)
    print(y)
    break

# model

import torch.nn as nn  # import_02 torch.nn


class LinearNet(nn.Module):  # 继承的源头 nn.Module （定义线性模型，给定输出）
    def __init__(self, n_feature):  # 输入为输入参数的数量
        super(LinearNet, self).__init__()  # 本class继承父系，初始化
        self.linear = nn.Linear(n_feature, 1)  # 新性质.linear变更，等于nn.Linear(num_in,num_out). 模型建立的关键步骤

        # forward 向前计算。。可是这里买你没有w和b啊

    def forward(self, x):  # 定义模型的计算，给x，回y
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)  # 使用print可以
#
# # 3种方案可替换上述net的构建
# # 1
net = nn.Sequential(nn.Linear(num_inputs, 1))  # 分层构建，只需要 一个nn Linear（in，out）。 上面的class方案不能用net【0】
# # 2
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # 3
# from collections import OrderedDict
#
# net = nn.Sequential(OrderedDict([('Linear', nn.Linear(num_inputs, 1))]))
print(net)
print(net[0])

# 查看可训练的参数。
# net定义中没有输入wb，但是依据线性特人性，知道w和b的维度
for param in net.parameters():  # 继承的特性net.parameters
    print(param)

# 初始化模型参数
from torch.nn import init  # 初始 import__03 from import.nn import init

# in suit 改变
init.normal_(net[0].weight, mean=0, std=0.01)  # 初始w，权重。背下来
init.constant_(net[0].bias, val=0)  # 初始b，截距，偏差

# 定义loss function. 这里是均方误差损失
loss = nn.MSELoss()  # MSELoss 均方误差损失（默认两个输入，yhat和y）
# 定义优化算法.小批量随机梯度下降(SGD)
import torch.optim as optim  # import_04 torch,optim

optimizer = optim.SGD(net.parameters(),
                      lr=0.03)  # 优化参数统一成为.parameters 和loss一样，是函数，此处是关于parameters的函数。optim.SGD（net。parameters，lr）
print(optimizer)

# 训练模型
num_epochs = 3
for e in range(1, num_epochs):
    for x, y in data_iter:
        output = net(x)  # yhat
        l = loss(output, y.view(-1, 1))  # loss
        optimizer.zero_grad()  # 清零
        l.backward()  # back
        optimizer.step()  # 更新
    print('epoch %d, loss: %f' % (e, l.item()))
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
