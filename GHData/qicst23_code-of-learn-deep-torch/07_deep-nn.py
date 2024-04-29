#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:16:58 2018

@author: jangqh
"""
from __future__ import division

import numpy as np
import torch
import mnist     ####本地模块

from torch import nn
from torch.autograd import Variable
'''
1、
'''
#从本地读取数据集
train_set = mnist.MNIST('./07_data', train = True, download = True)
test_set = mnist.MNIST('./07_data', train=False, download= True)

###### 这里读入的数据是PIL的格式，转换为numpy-array
i = 10  ##显示第i张图片
a_data, a_label = train_set[i]
print "刚读取出来的数据是PIL格式：",a_data
print "第{}张图片的标签：".format(i),a_label

a_data = np.array(a_data, dtype = 'float32')
print ("a_data shape:",a_data.shape)

print a_data
##0 表示黑色，1 表示黑色
#####################################

#第一层输入28*28 = 784  ，将数据变换为一维变量  reshape
def data_tf(x):
    x = np.array(x, dtype = 'float32') / 255
    x = (x-0.5) / 0.5       ###标准化
    x = x.reshape((-1,))    #拉平
    x = torch.from_numpy(x)
    return x

###重新载入数据集， 申明定义的数据变换
train_set = mnist.MNIST('./07_data', train = True, transform=data_tf, download=True)
test_set = mnist.MNIST('./07_data', train= False, transform=data_tf, download=True)

###
a, a_label = train_set[i]
print "第{}张图片的大小：".format(i), a.size()
print "第{}张图片的标签：".format(i), a_label
        
###
print "----------------------------"
from torch.utils.data import DataLoader
print "# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器...."
train_data = DataLoader(train_set, batch_size = 64, shuffle = True)
test_data = DataLoader(test_set, batch_size  =128, shuffle  =True)

a, a_label = next(iter(train_data))
print "一个"
print "一个批次的{train_data}的大小：", a.size()
print "一个批次的{train_label}的大小：", a_label.size()
"""
使用这样的数据迭代器是非常有必要的，如果数据量太大，就无法一次将他们全部读入内存，所以需要使用 python 迭代器，每次生成一个批次的数据
"""
##
print "----------------------------------"
###############################################3
print "使用Sequential 定义4层神经网络"
net = nn.Sequential(nn.Linear(784,400),
                    nn.ReLU(),
                    nn.Linear(400,200),
                    nn.ReLU(),
                    nn.Linear(200,100),
                    nn.ReLU(),
                    nn.Linear(100,10)
                    )

print "网络：", net

print "定义损失函数..."
criterion = nn.CrossEntropyLoss()
print "使用SGD..."
optimizer = torch.optim.SGD(net.parameters(), 1e-1)   


print "-----------------------"
print "开始训练...."
losses = []
acces  = []
eval_losses = []
eval_acces  = []


for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        
        #前向传播
        out = net(im)
        loss = criterion(out, label)
        
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #记录误差
        train_loss += loss.data[0]
        
        #计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc =num_correct / im.size()[0]
        train_acc += acc
    
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    
    ####在测试集上检验效果.....
    eval_loss = 0
    eval_acc = 0
    ######将模型改为预测模型###
    net.eval()
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        
        ###记录误差
        eval_loss += loss.data[0]
        
        #记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0]
        acc = num_correct / im.size()[0]
        eval_acc += acc
    
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print("epoch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}, Eval Loss:{:.6f}, Eval Loss:{:.6f}"
          .format(e, train_loss / len(train_data), train_acc / len(train_data), 
                  eval_loss /len(test_data), eval_acc / len(test_data)))
    
    
import matplotlib.pyplot as plt

####train loss 曲线
plt.plot(np.arange(len(losses)), losses)
plt.title("train loss")
plt.show()


####train acc 准确率曲线
plt.plot(np.arange(len(acces)), acces)
plt.title("train acc")
plt.show()

#####test loss 曲线
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()

##test acc 曲线
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()

    ###
    
    
    
        
        
        

    
    






















