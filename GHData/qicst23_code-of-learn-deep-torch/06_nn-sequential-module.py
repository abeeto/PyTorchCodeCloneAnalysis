#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:11:44 2017

@author: jangqh
"""

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


###
#def plot_decision_boundary(model, x, y):
    
    
    

    
##这次仍然处理二分类问题
np.random.seed(1)
m = 400     #样本数量
N = int(m/2)    #每一类的点数 200
D = 2   #维度
x = np.zeros((m,D))     #样本
y = np.zeros((m,1), dtype = 'uint8')     #label 向量, 0 红色  1蓝色
a = 4


###画出花瓣
for j in range(2):
    ix = range(N*j, N*(j+1))
    t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N) * 0.2  #theta
    r = a*np.sin(4*t)   #radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

#import matplotlib.pyplot as plt
#plt.scatter(x[:,0], x[:,1], c = y, s = 40, cmap = plt.cm.Spectral)
#plt.show()

"""
1、使用logistic 回归解决这个问题
"""
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))

optimizer = torch.optim.SGD([w, b], 1e-1)

def logistic_regression(x):
    return torch.mm(x, w) + 1

criterion = nn.SGD()

for e in range(100):
    out = logistic_regression(Variable(x))
    loss = criterion(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch:{}, loss:{}'.format(e+1, loss.data[0]))

    























