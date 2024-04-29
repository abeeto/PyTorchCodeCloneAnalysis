# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 16:42
# @Author  : zwenc
# @File    : Test.py

import torchvision.datasets    # 有很多成熟的数据集
import torchvision.transforms  # 可以选择的转换格式
import torch
from torch.utils.data import DataLoader # 对数据集操作
from torch.autograd import Variable     # 梯度操作
from torch import optim                 # 有优化算法，如梯度下降
import torch.nn as nn                   # 网络相关算法，如卷积，线性，relu等
from NetNN import Net                   # 用户自己的模块

batch_size = 50      # 每批数据的个数，DataLoader的参数
learning_rate = 0.01 # 学习速率
momentum = 0.9       # 惯性指数，用来更新权重
num_epoches = 20     # 迭代次数

# 获得训练数据，本地没有就会通过网络下载
otrain_data = torchvision.datasets.MNIST(
  root='./mnist/',
  train=True,
  transform=torchvision.transforms.ToTensor(),
  download=True,
)

# 获得测试数据，本地没有就会通过网络下载
otest_data = torchvision.datasets.MNIST(
  root='./mnist/',
  train=False,                                 # 下载测试集
  transform=torchvision.transforms.ToTensor(),  # 输出转换为Tensor类型
  download=True,                                # 如果本地没有，则下载
)

# 通过DataLoader函数将数据按照mini_bach，输出的train_data有data和Lable。Data的shape为[batch_size，1，28, 28]
# train_data = DataLoader(dataset=train_data,batch_size = batch_size,shuffle = True)
# test_data = DataLoader(dataset=test_data,batch_size = batch_size,shuffle = True)

# 图片集测试，输出图像的样子
# import matplotlib.pyplot as plt
# for D,L in train_data:
#     print(L[0])
#     plt.imshow(D[0][0])
#     plt.show()

# 定义模块
model = Net()

# 定义损失函数，其实就是一个计算公式
criterion = nn.CrossEntropyLoss()

# 定义梯度下降算法,把model内的参数交给他
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum = momentum)

for num_epoche in range(num_epoches):
    train_data = DataLoader(dataset=otrain_data,batch_size = batch_size,shuffle = True) # 打乱数据，处理格式
    model.train(mode=True)  # 设置为训练模式
    for index, (data,lable) in enumerate(train_data):
        # data torch.Size([60, 1, 28, 28])
        D = Variable(data,requires_grad=True).cuda()  # cuda表示放在GPU计算
        L = Variable(lable).cuda()                    # cuda表示放在GPU计算

        out = model(D)
        loss = criterion(out,L)  # loss 是一个值，不是向量
        optimizer.zero_grad()    # 清除上一次的梯度，不然这次就会叠加
        loss.backward()          # 进行反向梯度计算
        optimizer.step()         # 更新参数

    test_data = DataLoader(dataset=otest_data,batch_size = batch_size,shuffle = True) # 打乱数据顺序
    model.eval()  # 设置网络为评估模式
    eval_loss = 0 # 保存平均损失
    num_count = 0 # 保存正确识别到的图片数量
    for index, (data,lable) in enumerate(test_data):
        D = Variable(data).cuda()
        L = Variable(lable).cuda()

        out = model(D)
        loss = criterion(out,L)   # 计算损失，可以使用print输出
        eval_loss += loss.data.item() * L.size(0)  # loss.data.item()是mini-batch平均值

        pred = torch.max(out,1)[1] # 返回每一行中最大值的那个元素，且返回其索引。如果是0，则返回每列最大值
        num_count += (pred == L).sum() # 计算有多少个,这种方法只支持troch.tensor类型

    acc = num_count.float() / 10000
    eval_loss = eval_loss.float() / 10000
    print("num_epoche:%2d，num_count:%5d, acc: %6.4f, eval_loss:%6.4f"%(num_epoche,num_count,acc,eval_loss))






