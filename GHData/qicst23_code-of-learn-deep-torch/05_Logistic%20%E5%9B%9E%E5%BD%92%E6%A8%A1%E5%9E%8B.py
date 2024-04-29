#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:58:33 2018

@author: jangqh

"""
from __future__ import division  

import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
##随机种子
torch.manual_seed(2018)

#从data.txt中读取数据
with open('./05_data.txt', 'r') as f:
    data_list = [i.split('\n')[0] for i in f.readlines()]
#    print i.split('\n')[0]
#    print data_list[0].split(',')
#    data = [i for i in data_list]
    
    #str to float
    data = [(float(i.split(',')[0]), float(i.split(',')[1]), int(i.split(',')[2])) \
            for i in data_list]
#    print data
    
    print "------------------------"
    
#每一列的最大值
x0_max = max([i[0] for i in data])
print x0_max

x1_max = max([i[1] for i in data])
print x1_max

#归一化
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

#选择第一类点
x0 = list(filter(lambda x:x[-1] == 0,data))
print x0[:10]

print "==================\n"
#选择地二类点
x1 = list(filter(lambda x:x[-1] == 1,data))
print x1[:10]

####画散点图
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
    
plot_x1 = [i[0] for i in x1] 
plot_y1 = [i[1] for i in x1]
#    
#plt.plot(plot_x0, plot_y0, 'ro', label = 'x_0')
#plt.plot(plot_x1, plot_y1, 'bo', label = 'x_1')
#plt.legend(loc = 'best')
#plt.show()
##    


#转化成numpy的类型
np_data = np.array(data, dtype = 'float32')
x_data = torch.from_numpy(np_data[:,0:2]) ###每个data的前两列  大小为  100*2
y_data = torch.from_numpy(np_data[:,-1]).unsqueeze(1)
print "\n"
print type(y_data),y_data.size()
print y_data

#定义sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#画出sigmoid函数
plot_x = np.arange(-10, 10.01,0.01)
plot_y = sigmoid(plot_x)
    
#plt.plot(plot_x, plot_y, 'r')
#yy1 = [0,1.1]
#xx1 = [0,0]
#plt.plot(xx1, yy1, 'b')
#
#xx2 = [-11,11]
#yy2 = [0,0]
#plt.plot(xx2, yy2, 'g')
#plt.show()
#    

####导入动态图
x_data = Variable(x_data)
y_data = Variable(y_data)


import torch.nn.functional as F

##定义逻辑斯蒂回归模型
w = Variable(torch.randn(2, 1), requires_grad = True)
b = Variable(torch.zeros(1), requires_grad = True)


def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)

###画出参数更新之前的结果
w0 = w[0].data[0]
w1 = w[1].data[0]

b0 = b.data[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1
    
#plt.plot(plot_x, plot_y, 'g', label = 'cutting line')
#plt.plot(plot_x0, plot_y0, 'ro', label = 'x_0')
#plt.plot(plot_x1, plot_y1, 'bo', label = 'x_1')
#plt.legend(loc = 'best')
#plt.show()

####计算loss
def binart_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1-y)*(1-y_pred).clamp(1e-12).log()).mean()
    return -logits

print "x_data size:", x_data.size()
print "w size:", w.size()
print "b size:", b.size()

#zz = F.sigmoid(torch.mm(x_data, w)+b)
#print("x*w+b:",zz.size())
#print "b:",b

y_pred = logistic_regression(x_data)
print ("y_pred size:",y_pred.size())

loss = binart_loss(y_pred, y_data)
print "loss 1:",loss



##################
##################求导
loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data

y_pred = logistic_regression(x_data)
loss = binart_loss(y_pred, y_data)
print "loss 2:",loss

"""
使用torch.optim 更新参数
"""
print "------------------我是分割线-----------------"
str = """
上面的参数更新方式其实是繁琐的重复操作，如果我们的参数很多，比如有 100 个，那么我们需要写 100 行来更新参数，为了方便，我们可以写成一个函数来更新，其实 PyTorch 已经为我们封装了一个函数来做这件事，这就是 PyTorch 中的优化器 torch.optim

使用 torch.optim 需要另外一个数据类型，就是 nn.Parameter，这个本质上和 Variable 是一样的，只不过 nn.Parameter 默认是要求梯度的，而 Variable 默认是不求梯度的

使用 torch.optim.SGD 可以使用梯度下降法来更新参数，PyTorch 中的优化器有更多的优化算法，在本章后面的课程我们会更加详细的介绍

将参数 w 和 b 放到 torch.optim.SGD 中之后，说明一下学习率的大小，就可以使用 optimizer.step() 来更新参数了，比如下面我们将参数传入优化器，学习率 lr 设置为 1.0

"""

print str
from torch import nn
w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))

def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)

optimizer = torch.optim.SGD([w, b], lr = 1)

import time 

start = time.time()
for e in range(300):
    # 前向传播
    y_pred = logistic_regression(x_data)
    loss = binart_loss(y_pred, y_data)
    
    #后向传播
    optimizer.zero_grad()     ##使用优化器将梯度归0
    loss.backward()
    optimizer.step()    #使用优化器更新参数
    
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.size()[0]
    
    if (e+1) %100 == 0:
        print ('epoch:{}, Loss :{:.5f}, Acc:{:.5f}'.format(e+1, loss.data[0], acc))
        
during_time = time.time() - start
print ()
print ("During Time : {:.3f} s".format(during_time))
    

str = """
可以看到使用优化器之后更新参数非常简单，只需要在自动求导之前使用optimizer.zero_grad() 来归 0 梯度，然后使用 optimizer.step()来更新参数就可以了，非常简便

同时经过了 1000 次更新，loss 也降得比较低了
"""
print str
print "-----------我是分割线---------------"
#print(y_pred.size())
#mask = y_pred.ge(0.5).float()
##print mask
##print y_data
#acc = (mask == y_data).sum().data[0]
#print acc
#print y_data.size()[0]
#
##print acc
##
##for i in range(10):
##    print "y_data {}:".format(i), y_data[i], "y_pred {}:".format(i), y_pred[i],
#    
    

###########################
    
w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0].numpy().astype(float)
print "w0:", w0
print "w1:", w1
print "b0:", b0

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x -b0) / w1

plot_x = plot_x.tolist()
plot_y =plot_y.tolist()

print "type plot_x:",type(plot_x[0]), " ", plot_x[0]
print "type plot_y:",type(plot_y[0]), ' ', plot_y[0]
print "type plot_x0:",type(plot_x0[0]), ' ', plot_x0[0]
print "type plot_y0:",type(plot_y0[0]), ' ', plot_y0[0]
print "type plot_y1:",type(plot_y1[0]), ' ', plot_y1[0]


         
#import matplotlib.pyplot as plt
#plt.plot(plot_x, plot_y, 'g', label = "cutting line")
#plt.plot(plot_x0, plot_y0, 'ro', label = 'x_0')
#plt.plot(plot_x1, plot_y1, 'bo', label = 'x_1')
#plt.legend(loc = 'best')
#plt.show()
    

"""
使用自带的loss
"""   

criterion = torch.nn.CrossEntropyLoss()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     