#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:44:36 2018

@author: jangqh
"""
import sys
sys.path.append('..')



import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms as tfs
import mnist

##定义数据
data_tf  =  tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5], [0.5])
        ])   ###标准化

train_set = mnist.MNIST('./07_data', train = True, transform = data_tf,download = True)
test_set = mnist.MNIST('./07_data', train = False, transform = data_tf,download = True)

train_data = DataLoader(train_set, 64, True, num_workers=4)
test_data = DataLoader(test_set, 64, True, num_workers=4)

i = 10
a, a_label = train_set[i]
print "第{}张图片的大小：".format(i), a.size()
print "第{}张图片的标签：".format(i), a_label

###定义模型
class rnn_classify(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)   ###使用两层lstm
        self.classifier = nn.Linear(hidden_feature, num_layers)   ##将最后一个rnn的输出
                                                                ##使用全连接得到最后的分类结果
    def forward(self, x):
        """
        x 为大小为 （batch， 1, 28, 28）所以我们需要将其转换为RNN的输入形式，即（28， batch，28） 
        """        
        x = x.squeeze()  ##去掉（batch， 1， 28， 28）中的1，变成（batch， 28，28）
        x = x.permute(2, 0, 1)   ###将最后一维放到第一维
        out, _ = self.rnn(x)    ##使用默认的隐藏状态， 得到的是out是（28，batch，hidden_feature）
        out = out[-1,:,:]  ##取序列的最后一个  大小是（batch, hidden_feature）
        out = self.classifier(out)  ##得到分类结果
        return out

net  =  rnn_classify()
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

optimizer  =  torch.optim.Adadelta(net.parameters(), 1e-1)


##开始训练
from utils import train
train(net, train_data, test_data, 10, optimizer, criterion)





