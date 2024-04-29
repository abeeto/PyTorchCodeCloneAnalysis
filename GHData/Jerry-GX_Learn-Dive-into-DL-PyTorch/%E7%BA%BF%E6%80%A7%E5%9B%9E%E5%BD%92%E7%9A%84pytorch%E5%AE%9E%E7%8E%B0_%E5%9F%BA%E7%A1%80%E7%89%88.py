# -*- coding: utf-8 -*-
"""
只利用Tensor和autograd实现一个线性回归的训练
"""

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 生成数据集
# y = Xw + b + ε
# X（2000*2） w =(1.8, -3.9).T b = 6.6 ε = noise
feature_num = 2
sample_num = 2000
true_w = [1.8, -3.9]
true_b = 6.6
samples = torch.rand(sample_num, feature_num, dtype = torch.float32)
data = true_w[0]*samples[:,0] + true_w[1]*samples[:,1] + true_b
noise = np.random.normal(0, 0.01, size=data.size())
data = data + torch.tensor(noise)

#绘图
def use_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

#观察data与feature的线性关系    
#set_figsize()
#plt.scatter(samples[:,0].numpy(), data.numpy(), 0.5)

#数据读取 定义一个函数：它每次返回batch_size（批量大小）个随机样本的feature和data
def data_iter(batch_size, features, data):
    sample_num = len(features)
    index = list(range(sample_num))
    random.shuffle(index)  #序列元素随机排序
    for i in range(0, sample_num, batch_size):
        j = torch.LongTensor(index[i: min(i + batch_size, sample_num)])
        yield  features.index_select(0, j), data.index_select(0, j)  
        #index_select(0,j) 0表示按行索引 1表示按列索引 j是一个tensor，为索引序号
        
'''batch_size = 10
for X, y in data_iter(batch_size, samples, data):
    print(X, y)
    break'''

#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (feature_num, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#定义模型 linear regression
def linreg(X, w, b):
    return torch.mm(X, w)+b

#定义loss function
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2 /2

#定义优化算法（小批量随机梯度下降）
def sgd(params, lr, batch_szie):
    for param in params:
        param.data -= lr*param.grad / batch_size
        
#训练模型
lr = 0.01
num_epochs = 60
batch_size = 10
net = linreg
loss = squared_loss
loss_display = []

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, samples, data):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(samples, w, b), data)
    loss_display.append(train_l.mean().item())  #item() 显示元素
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
       
#可视化训练过程
set_figsize()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(1,num_epochs+1), loss_display)

print(true_w, '\n', w)
print(true_b, '\n', b)
















