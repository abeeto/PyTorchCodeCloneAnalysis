#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:33:35 2018

@author: jangqh
"""
from __future__ import division
import numpy as np
import torch

import mnist
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

####数据的预处理 归一化等
def data_tf(x):
    x = np.array(x, dtype = 'float32') / 255   ###将数据归到0-1之间
    x = (x - 0.5) / 0.5   ##标准化
    x = x.reshape((-1, ))   ##拉平
    x = torch.from_numpy(x)
    return x


######读取数据  并转换
train_set = mnist.MNIST('./data', train = True, transform=data_tf, download=True)  
test_set  = mnist.MNIST('./data', train = False, transform=data_tf, download=True)

###定义loss
criterion = nn.CrossEntropyLoss()


###使Sequential 定义3层神经网络
net = nn.Sequential(
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200,10),
        )
#

""""
实现随机梯度下降法
"""
def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr*param.grad.data
###################################
#print "设置batch_size = 1"

#train_data = DataLoader(train_set, batch_size = 1, shuffle = True)
#

#print "开始训练...."
#losses1 = []
#idx = 0
#
#start = time.time()####计时开始
#for e in range(5):
#    train_loss = 0
#    for im, label in train_data:
#        im = Variable(im)
#        label = Variable(label)
#        
#        ##向前传播
#        out = net(im)
#        loss = criterion(out, label)
#        
#        ##反向传播
#        net.zero_grad()   ##梯度归零
#        loss.backward()
#        sgd_update(net.parameters(), 1e-2)   ##使用0.01的学习率
#        
#        ###记录误差
#        train_loss += loss.data[0]
#        if idx % 30 == 0:
#            losses1.append(loss.data[0])
#        idx += 1
#    print ('epoch:{},Train Loss :{:.6f}'.format(e, train_loss / len(train_data)))
#end = time.time()  ####计时结束
#print('使用时间：{:.5f} s'.format(end-start))
#
#####画图 
#x_axis = np.linspace(0.5, len(losses1), endpoint = True)
#plt.semilogy(x_axis, losses1, label = 'batch_szie  = 1')
#plt.legend(loc = 'best')
#
#print """
#可以看到，loss 在剧烈震荡，因为每次都是只对一个样本点做计算，每一层的梯度都具有很高的随机性，而且需要耗费了大量的时间
#"""


##########
#train_data = DataLoader(train_set, batch_size=256, shuffle=True)
#print "设置 batch_szie = 64"
#
#net = nn.Sequential(
#        nn.Linear(784, 200),
#        nn.ReLU(),
#        nn.Linear(200, 10),
#        )
#
#print "开始训练...."
#losses2 = []
#idx = 0
#start = time.time()  ##开户四计时
#for e in range(5):
#    train_loss = 0
#    for im ,label in train_data:
#        im = Variable(im)
#        label = Variable(label)
#        
#        ##前向传播
#        out = net(im)
#        loss = criterion(out, label)
#        
#        ###反向传播
#        net.zero_grad()
#        loss.backward()
#        sgd_update(net.parameters(), 1e-2)
#        
#        ###记录误差
#        train_loss += loss.data[0]
#        if idx % 30 == 0:
#            losses2.append(loss.data[0])
#        idx += 1
#    print ('epoch:{}, Train Loss:{:.6f}'.format(e, train_loss / len(train_data)))
#
#end = time.time()
#print("使用时间：{:.5f} s".format(end - start))
#
#x_axis = np.linspace(0, 5, len(losses2), endpoint = True)
#plt.semilogy(x_axis, losses2, label = 'batch_szie = 64')
#plt.legend(loc = 'best')
#plt.show()
#
#print """
#通过上面的结果可以看出，loss没有batch等于1震荡的那么厉害，同时也可以降到一定的程度，
#时间上比之前快乐很多，因为按照batch的数量计算上更快，同时梯度对比于batch—size=1，的情况也更接近真实梯度，所以batch size越大，梯度越稳定，batch size越小，随机性越大，，batch size越大，对内存的要求越高，同时不利于网络跳出局部极小值，所以通常使用基于batch的随机梯度下降法，batch size的大小根据实际情况选择
#
#"""


print"""
实际上我们并不用自己造轮子，因为 pytorch 中已经为我们内置了随机梯度下降发，而且之前我们一直在使用，下面我们来使用 pytorch 自带的优化器来实现随机梯度下降
"""

train_data = DataLoader(train_set, batch_size = 64, shuffle=True)

net = nn.Sequential(
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,10),
        )

optimizer = torch.optim.SGD(net.parameters (), 1e-2)

###
print "开始训练..."

start = time.time()
print start

for e in range(5):
    train_loss = 0
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        
        ###前向传播
        out = net(im)
        loss = criterion(out, label)
        
        ###反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ##记录误差
        train_loss += loss.data[0]
    print("epoch :{}, Train loss:{:>6f}".format(e, train_loss / len(train_data)))
end  = time.time()

print ("使用时间：{:.5f}".format(end- start))


        
        
    
    
    
    
    
    
    
    
    
    
    
    
    













































