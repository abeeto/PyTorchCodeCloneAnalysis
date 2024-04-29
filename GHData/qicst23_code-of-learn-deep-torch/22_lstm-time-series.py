#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:45:11 2018

@author: jangqh
"""
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_csv = pd.read_csv('./data.csv', usecols = [1])

#plt.plot(data_csv)

###数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values

dataset = dataset.astype('float32')
print dataset,"len:",len(dataset)
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
print max_value,min_value
dataset = list(map(lambda x :x / scalar, dataset))
print dataset


###创建数据集
def create_dataset(dataset, look_back = 2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

##创建好输入输出
data_X, data_Y = create_dataset(dataset)
print "len data_X:",len(data_X)
print "len_data_Y:",len(data_Y)

train_size = int(len(data_X) * 0.7)
print train_size
test_size = len(data_X) - train_size
print test_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

###RNN读入的数据（seq,batch,feature）
import torch
print train_X.shape
train_X = train_X.reshape(-1, 1, 2)
print train_X.shape
train_Y = train_Y.reshape(-1, 1, 1)
print train_Y.shape

test_X  =test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

from torch import nn
from torch.autograd import Variable

###定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_szie = 1, num_layers = 2):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_szie)
    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

net = lstm_reg(2, 4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adadelta(net.parameters(), lr  =1e-1)


###开始训练
for e in range(5000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    
    ##前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    
    ###反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(e + 1) % 100 == 0:
        print ('Epoch:{}, Loss:{:.5f}'.format(e+1, loss.data[0]))

#############测试模式
net = net.eval()   ###转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data)   ####测试集的预测结果

##改变输入的格式
pred_test = pred_test.view(-1).data.numpy()

###画出实际结果与预测结果
plt.plot(pred_test, 'r', label = 'prediction')
plt.plot(dataset, 'b', label  = 'real')
plt.legend(loc = 'best')










