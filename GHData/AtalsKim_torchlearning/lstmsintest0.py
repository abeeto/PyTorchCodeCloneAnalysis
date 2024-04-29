# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:08:52 2019

@author: 64451
"""

# https://blog.csdn.net/hustchenze/article/details/78696771


import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import xlrd







def loaddata (xlpath):

    # 打开文件

    workbook = xlrd.open_workbook(xlpath)
    # 获取所有sheet
    print(workbook.sheet_names()) # [u'sheet1', u'sheet2']
    sheet2_name = workbook.sheet_names()[0]
    sheet2 = workbook.sheet_by_index(0)
    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)
    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(0)  # 获取第四行内容
    # 9 左高低，11 左轨向   10右高低   12右轨向
    inputs = sheet2.col_values(9, start_rowx=1)
    targets = sheet2.col_values(11, start_rowx=1)
    print('Datasetlens: ', len(inputs))
    return inputs, targets


def SeriesGen(N):
    x = torch.arange(0,N,0.01)
    return x, torch.sin(x)
 
def trainDataGen(x, seq, k):
    """

    :param x: input
    :param seq: output
    :param k: 每个batch的大小
    :return: 列表 数组
    数据长度-k-1 batch个数
    """
    dat = list()
    L = len(seq)
    # k 其实就是训练集的长度单元，
    # 多个训练集，[[x1,x2....],[y1y2]]
    for i in range(L-k-1):
        indat = x[i:i+k]
        outdat = seq[i:i+k]
        dat.append((indat, outdat))
    return dat
 
def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)
 
#def newsinedata(n):
#dat = list()
#for i in range(n):
#    indat = torch.tensor(i)
#    outdat = torch.sin(np.pi*indat)

# Test
# x, y = SeriesGen(20)
# dat = trainDataGen(x.numpy(), y.numpy(),600)
 






class LSTMpred(nn.Module):
 
    def __init__(self,input_size,hidden_dim):
        super(LSTMpred,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size,hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim,1)
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
 
    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq),-1))
        return outdat

model = LSTMpred(1,8).to('cuda')
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
 




# 配置输入batch
xlpath = r'C:\Users\64451\Desktop\2019-07-23基于XX学习的轨道不平顺生成方法 基于非IFFT的轨道不平顺幅值生成方法\SIX\BJ_GJC_W_6号线上行2014010601.xlsx'
x, y = loaddata(xlpath)
dat = trainDataGen(x, y, 1000)



num_epochs = 100
for epoch in range(num_epochs):
    print(epoch)
    loss = 0
    for seq, outs in dat:
        seq = ToVariable(seq).cuda()
        seq = seq.view(len(seq), 1)
        outs = ToVariable(outs).cuda()
        outs = seq.view(len(outs), 1)

        # 由于输入的时array 改为 a x 1 的格式
        # 修改完之后有明显降低
        #outs = torch.from_numpy(np.array([outs]))

        # 清除网络状态
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        modout = model(seq).cuda()
        loss = loss_function(modout, outs)
        # 反向传播求梯度
        loss.backward()
        # 更新参数
        optimizer.step()
    # 放里头一直更新
    if epoch % 1 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch, num_epochs, loss.item()))
              
        
        
 
predDat = []
for seq, trueVal in dat[:]:
    seq = ToVariable(seq).cuda()
    trueVal = ToVariable(trueVal).cuda()
    x = x.cuda()
    model = model.cuda()
    data = model(seq)[-1].data.cpu()
    predDat.append(data.numpy()[0])
 
 
fig = plt.figure()
plt.plot(y.numpy())
plt.plot(predDat)
plt.show()
