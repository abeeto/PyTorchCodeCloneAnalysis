# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:08:52 2019

@author: 64451
"""

# https://blog.csdn.net/hustchenze/article/details/78696771
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import xlwt
import os
import tkinter as tk
import tkinter.filedialog
import sys


def postPlot(model, x, y):

    root = tk.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename(filetypes=[('NET_files', '.pkl')])

    checkpoint = torch.load(filepath)
    # 重新构造
    # 可能有点问题
    NEUNUM = len(checkpoint['lstm.weight_hh_l0'][0])
    tmodel = LSTMpred(1, NEUNUM).to('cuda')
    tmodel.load_state_dict(checkpoint)
    # plot 都在cpu空间
    testx = ToVariable(x).to('cuda')
    predDat = tmodel(testx).data.to('cpu').numpy()
    simplot(y, predDat)
    return y, predDat



def simplot(trueDat, predDat):
    fig = plt.figure()
    # plt.plot(y.numpy())
    plt.plot(trueDat, label='Turedata')
    plt.plot(predDat, label='Predict', alpha=0.7)
    plt.legend()
    plt.show()
    return None

def save2excel(data, xlname='Pred_Truth.xls'):
    """
    data: [array(predict),array(Truth)]
    """
    
    xlsfilename = xlname
    workbook = xlwt.Workbook(encoding = 'utf-8')
    wsheet = workbook.add_sheet('Test', cell_overwrite_ok=True)
    for j in range(len(data)):
        for i in range(len(data[j])):    
            wsheet.write(i,j, label = float(data[j][i]))
    workbook.save(xlsfilename)
    print("Excel out finished.")
    print(os.path.abspath(xlname))
    return None


def loaddata (xlpath, length=-1, start=1):

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
    if length == -1:
        inputs = sheet2.col_values(9, start_rowx=start)
        targets = sheet2.col_values(11, start_rowx=start)
    else:
        inputs = sheet2.col_values(9, start_rowx=start, end_rowx=start+length)
        targets = sheet2.col_values(11, start_rowx=start, end_rowx=start+length)
    print('Datasetlens: ', len(inputs))
    return inputs, targets


def SeriesGen(N):
    x = torch.arange(0, N, 0.01)
    return x, torch.sin(x)
 
    
def trainDataGen(x, seq, k, step = 10*4):
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
    num = 0
    for i in range(0,L-k-1, step):
        indat = x[i:i+k]
        outdat = seq[i:i+k]
        dat.append((indat, outdat))
#        print('TrainData: length ', len(dat))
#        print(num)
        num += 1
    print('Batch Number:',len(dat))
    print('Batch size:',len(dat[0][0]))
    return dat
 
    
def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)
 
 


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

TESTMOD = False
NEUNUM = 10
model = LSTMpred(1, NEUNUM).to('cuda')
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 配置输入batch
xlpath = r'excelTest37000.xlsx'
# 12500~ 15000 有坏值
x, y = loaddata(xlpath, length = -1, start=1)
# 后1000 留下测试
testlen = 2000
step = 40
dat = trainDataGen(x, y, testlen, step=step)
num_epochs = 500
rloss = []
print("NNumber:{}, TestLen:{}, Epochs:{}".format(NEUNUM, testlen, num_epochs))


# 测试模块， 会假死
if TESTMOD:
    tlen = 10000
    tstart = 10000
    x = x[tstart:tstart+tlen]
    y = y[tstart:tstart+tlen]
    trueDat, predDat = postPlot(model, x, y)
    save2excel([trueDat, predDat], xlname='Pred_Truth2.xls')
    quit()


print("Epoch Start...")
for epoch in range(num_epochs):
    print(epoch)
    running_loss = 0.0
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
        # statistics
        running_loss += loss.item()
    # 临时绘图

    testx = ToVariable(x[-testlen:]).cuda()
    predDat = model(testx).data.cpu()
    predDat = predDat.numpy()
    trueDat = y[-testlen:]
    simplot(trueDat, predDat)
    rloss.append(running_loss)
    print('Epoch[{}/{}], loss:{:.6f}'.format(epoch, num_epochs, running_loss))

    # 保存模型参数
    # 保存模型
    if epoch % 10 == 0:
        pklname = 'param_N{}_Len{}_Ep{}_St{}.pkl'.format(NEUNUM,testlen,num_epochs,step)
        torch.save(model.state_dict(), pklname)
        print("> Parameters have been saved.")
              


# 最终测试
predDat = model(ToVariable(x[-testlen:]).cuda()).data.cpu()
predDat = np.array(predDat)
trueDat = y[-testlen:]


fig = plt.figure()
plt.plot(trueDat, label= 'Turedata')
plt.plot(predDat, label= 'Predict', alpha=0.4)
plt.legend()
plt.show()
# 保存至EXCEL
save2excel([trueDat, predDat], xlname='Pred_Truth2.xls')






# torch.save(model, 'net.pkl') # 全部网络











# 挑选新的测试集
#x2, y2 = loaddata(xlpath, length = 5000, start = 20000)
## 后1000 留下测试
#testlen = 500
#dat = trainDataGen(x2, y2, testlen)
#predDat2 = []
#for seq, trueVal in dat:
#    seq = ToVariable(seq).cuda()
#    trueVal = ToVariable(trueVal).cuda()
##    x = x.cuda()
#    model = model.cuda()
#    data = model(seq)[-1].data.cpu()
#    predDat2.append(data.numpy()[0])
# 
## 输出
#save2excel([y2[:-501], predDat2], xlname='Pred_Truth2.xls')
