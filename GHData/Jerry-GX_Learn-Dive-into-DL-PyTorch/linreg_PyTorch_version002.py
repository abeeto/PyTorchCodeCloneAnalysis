# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:13:31 2019

@author: GX
"""

import torch
import torch.utils.data as Data  #PyTorch提供data包读取数据。将导入的data模块用Data代替
import torch.nn as nn
from torch.nn import init
from matplotlib import pyplot as plt
import numpy as np

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


#数据读取模块
batch_size = 10
dataset = Data.TensorDataset(samples, data)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

'''
for X, y in data_iter:   #数据读取测试
    print(X,y)
    break
'''
#定义模型
net = nn.Sequential(
    nn.Linear(feature_num, 1)
    # 此处还可以传入其他层
    )
'''
参数：
in_features - 每个输入样本的大小
out_features - 每个输出样本的大小
bias - 如果设置为False，则图层不会学习附加偏差。默认值：True
'''

'''
for param in net.parameters():
    print(param)
'''

#初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

#Loss function
loss = nn.MSELoss()

#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

#模型训练
epoch_num = 15
loss_display = []
for epoch in range(1, epoch_num+1):
    for X,y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1,1).float())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    loss_display.append(l.mean().item())
    print('epoch: %d, loss: %f'%(epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
plt.title("Training Result")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(1,epoch_num+1), loss_display)







