#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:12:30 2018

@author: jangqh
"""
from __future__ import division
import numpy as np
import torch
#from torchvision.data import MNIST
import mnist ####导入本地模块
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

import time
import matplotlib.pyplot as plt




print """
动量法:
相当于每次在进行参数更新的时候，都会将之前的速度考虑进来，每个参数在各个方向上的移动幅度
不仅取决于当前的速度，还取决于过去各个梯度在各个方向上是否一致，如果一个梯度一直沿着当前方向更新，
那么更新幅度会越来与大，如果一个梯度在一个方向上不断变化，那么其更新幅度就会被衰减，这样我们就可以
使用一个较大的学习律，使得收敛更快，同时梯度比较大的方向就会因为动量的关系每次更新的幅度减少
------------------------------------
"""




#######实现一个动量法
def sgd_momentum(parameters, vs, lr, gamma):
    for param, v in zip(parameters, vs):
        v[:] = gamma * v + lr * param.grad.data[0]
        param.data = param.data - v


########################数据预处理
def data_tf(x):
    x = np.array(x, dtype = 'float32') / 255
    x = (x- 0.5) / 0.5   ###标准化
    x = x.reshape((-1,))  ##拉平
    x = torch.from_numpy(x)
    return x

####读取数据
train_set  =  mnist.MNIST('./data', transform=data_tf, download=True)
test_set = mnist.MNIST('./data', transform=data_tf, download=True)

###定义loss函数
criterion  =  nn.CrossEntropyLoss()


#####
train_data = DataLoader(train_set, batch_size = 64, shuffle = True)

###使用Sequential定义三层神经网络
net = nn.Sequential(
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
        )


###将速度初始化为参数相同的零向量
vs = []
for param in net.parameters():
    print param.size()
#    vs.append(torch.zeros_like(param.data))
    
####
print """
开始训练...
"""
losses = []
start = time.time()

idx = 0
#optimizer = torch.optim.SGD(net.parameters(), lr = 1e-2, momentum=0.9)   #####加动量
#optimizer = torch.optim.Adagrad(net.parameters(), lr = 1e-2)  ###加Adagrad
#optimizer = torch.optim.RMSprop(net.parameters(), lr = 1e-3, alpha=0.9)   ###RMSprop
#optimizer = torch.optim.Adadelta(net.parameters(), rho = 0.9)  ###Adadelta  
optimizer =  torch.optim.Adam(net.parameters(), lr = 1e-3)
  
for e in range(5):
    train_loss = 0
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        
        ####前向传播
        out = net(im)
        loss = criterion(out, label)
        
        
        ##反向传播
        net.zero_grad()
        loss.backward()
#        sgd_momentum(net.parameters(), vs,1e-2,0.9)   ###使用动量参数为0.9，学习率0.01
        optimizer.step()
                     
        ###记录误差
        train_loss += loss.data[0]
        
        if idx % 30 == 0:
            losses.append(loss.data[0])
        idx += 1
        
        losses.append(loss.data[0])
    print ('epoch:{},Train Loss:{:.6f}'.format(e, train_loss / len(train_data)))
end = time.time()
print ("s使用的时间：{:.5f}".format(end-start))

x_axis = np.linspace(0, 5, len(losses), endpoint = True)
plt.semilogy(x_axis, losses, label = "momentum:0.9")
plt.show()













































