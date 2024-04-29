#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:24:44 2018

@author: jangqh
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

"""
对于RNN,两种调用方式
1、torch.nn.RNNCell()：只能接受序列中的单步的输入，必须传入隐藏状态
参数:input_size, hidden_size, bias, nonlinearity
2、torch.nn.RNN():接受一个序列的输入，默认出入全0的隐藏状态，也可以申明隐藏的状态
参数:input_size表示输入的x的特征维度
     hidden_size:表示输出的特征维度
     num_layers:表示网络的层数
     nonlinearity:表示非线性激活函数，默认tanh
     bias:表示使用偏置，默认使用
     batch_first:表示输入的数据形式，默认False，就是将序列的长度放在第一位，batch放在第二位
     dropout：表示是否输出应用曾dropout
     bidirectional：表示是否使用双向的rnn,默认False
"""

#定义一个单步的rnn
rnn_single = nn.RNNCell(input_size = 100, hidden_size=200)

#访问其中的参数
print rnn_single.weight_hh
print rnn_single.weight_ih


##构造一个序列，长度为6，batch为5，特征为100
x = Variable(torch.randn(6,5,100))

#定义初始的记忆状态
h_t = Variable(torch.zeros(5,200))

#传入rnn
out = []
for i in range(6):  #通过循环6次作用在整个序列上
    h_t = rnn_single()




























