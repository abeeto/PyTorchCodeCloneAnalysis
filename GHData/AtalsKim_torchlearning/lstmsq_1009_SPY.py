# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:55:54 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
__author__ = 'Atlas'
'''
Author: Atlas Kim
Description: 
Modification: 
'''

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:49:54 2019

@author: Administrator
"""

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
import visdom

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


# 用于读取原始数据，继续计算
def load_net_state(netori, total=0):
    """

    :param netori: 输入的model
    :param total: 1 表示导入网络结构，0表示只导入参数
    :return model 一个神经网络结构，好像默认是cuda
    """

    def uigetpath(fileend='.pkl'):
        import tkinter as tk
        import tkinter.filedialog
        root = tk.Tk()
        root.withdraw()
        filepath = tkinter.filedialog.askopenfilename(filetypes=[('NET_files', fileend)])
        return filepath

    if total == 1:
        # 整体
        netori = torch.load(uigetpath(fileend='.pkl'))
    else:
        # 现有模型的参数
        netori.load_state_dict(torch.load(uigetpath))

    return netori


# 单维最大最小归一化
def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i] - low) / delta
    return list, low, high


# 反归一化函数
def FNoramlize(list, low, high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i] * delta + low
    return list


def postPlot(model, x, y):
    root = tk.Tk()
    root.withdraw()
    filepath = tkinter.filedialog.askopenfilename(filetypes=[('NET_files', '.pkl')])

    checkpoint = torch.load(filepath)
    # 重新构造
    # 可能有点问题
    NEUNUM = len(checkpoint['lstm.weight_hh_l0'][0])
    tmodel = LSTMpred(1, NEUNUM).to(device)
    tmodel.load_state_dict(checkpoint)
    # plot 都在cpu空间
    testx = ToVariable(x).to(device)
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
    workbook = xlwt.Workbook(encoding='utf-8')
    wsheet = workbook.add_sheet('Test', cell_overwrite_ok=True)
    for j in range(len(data)):
        for i in range(len(data[j])):
            wsheet.write(i, j, label=float(data[j][i]))
    workbook.save(xlsfilename)
#    print("Excel out finished.")
#    print(os.path.abspath(xlname))
    return None


def loaddata(xlpath, length=-1, start=1):
    # 打开文件
    workbook = xlrd.open_workbook(xlpath)
    # 获取所有sheet
    print(workbook.sheet_names())  # [u'sheet1', u'sheet2']
    sheet2_name = workbook.sheet_names()[0]
    sheet2 = workbook.sheet_by_index(0)
    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)
    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(0)  # 获取第四行内容
    # 9 左高低，11 左轨向   10右高低   12右轨向
    if length == -1:
        inputs = sheet2.col_values(10, start_rowx=start)
        targets = sheet2.col_values(12, start_rowx=start)
    else:
        inputs = sheet2.col_values(10, start_rowx=start, end_rowx=start + length)
        targets = sheet2.col_values(12, start_rowx=start, end_rowx=start + length)
    print('Datasetlens: ', len(inputs))
    return inputs, targets


def SeriesGen(N):
    x = torch.arange(0, N, 0.01)
    return x, torch.sin(x)


def trainDataGen(x, seq, k, step=10 * 4):
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
    for i in range(0, L - k - 1, step):
        indat = x[i:i + k]
        outdat = seq[i:i + k]
        dat.append((indat, outdat))
        #        print('TrainData: length ', len(dat))
        #        print(num)
        num += 1
    print('Batch Number:', len(dat))
    print('Batch size:', len(dat[0][0]))
    return dat


def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.0f' % (total / 1))
    return total


class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim, batchsize, num_layer=1):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.batchsize = batchsize
        # self.hidden = self.init_hidden()
        # 数据归一化操作
        # self.bn1 = nn.BatchNorm1d(num_features=320)
        # 增加DROPout 避免过拟合
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layer, dropout=0.5)
        # outfeature = 1
        self.hidden2out = nn.Linear(self.hidden_dim, 1)

    # 第一个求导应该不用的吧
    def init_hidden(self):
        return (
            Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).to(
                device),
            Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).to(
                device))

    def forward(self, seq):
        # 三个句子，10个单词，1000

        # hc维度应该是 [层数，batch, hiddensize]
        # out 维度应该是[单词, batch, hiddensize]
        # lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        # seq =1  batch 1 vec 200

        # vecinput 行数据的个数
        # input >>> [seq_len, batchsize, input_size]
        # out >>> [seq_len, bathchsize, hiddenlayernum]
        # h,c >>> [层数，batchsize, hiddensize]
        lstm_out, self.hidden = self.lstm(
            seq.view(int(len(seq) / self.batchsize), self.batchsize, 1), self.hidden)
        # 是不是多对一的话留下最后结果
        # outdat = self.hidden2out(lstm_out[-1].view(self.batchsize, -1))
        # return outdat.view(-1)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat


def main(NEUNUM = 16,NLAYER = 4,testlen = 1000,num_epochs = 10000,caseN = 9999):
    TESTMOD = False
    LOADPKL = False
    # batchsize
    step = 4
    port = 6007


    visdomenv = 'PytorchTest%d' % caseN
    # inputsize 这个应该是指特征的维度，所以是1

    # 模型构建
    if LOADPKL:
        model = load_net_state(1, 1)
    else:
        model = LSTMpred(1, NEUNUM, testlen, NLAYER).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    mparanum = print_model_parm_nums(model)

    # 数据读取
    xlpath = r'Prep_BJ_GJC_W2014010602.xlsx'
    x, y = loaddata(xlpath, length=80000, start=1)
    trainlen = round(len(x) * 0.8)
    dat = trainDataGen(x[:trainlen], y[:trainlen], testlen, step=step)
    testdata = trainDataGen(x[trainlen:], y[trainlen:], testlen, step=step)
    # 模型计算总参数输出
    rloss = []
    Tloss_list = []
    print(
        "NNumber:{}, TestLen:{}, Epochs:{}, Nlayer:{}, paranum:{}".format(NEUNUM, testlen,
                                                                          num_epochs,
                                                                          NLAYER,
                                                                          mparanum))

    # 测试模块， 会假死
    if TESTMOD:
        tlen = 10000
        tstart = 10000
        x = x[tstart:tstart + tlen]
        y = y[tstart:tstart + tlen]
        trueDat, predDat = postPlot(model, x, y)
        save2excel([trueDat, predDat], xlname='Pred_Truth2.xls')
        quit()

    # seqn, feature1,
    # batch 一次送多少个数据

    running_loss = 0.0
    # 使用VISDOM 进行绘图
    vis = visdom.Visdom(env=visdomenv, port=port)
    vis.text(
        "NNumber:{}, TestLen:{}, Epochs:{}, Nlayer:{}, paranum:{}".format(NEUNUM, testlen,
                                                                          num_epochs,
                                                                          NLAYER,
                                                                          mparanum),
        win='Training set')

    # 开始训练
    #    _ = input('Press any key to continue.......')
    print("Epoch Start...")
    for epoch in range(num_epochs):

        #        # 感觉目标应该是psd的相差较小
        #        seq = 2 * abs(np.fft.fft(x)) ** 2 / (len(x))
        #        outs = 2 * abs(np.fft.fft(y)) ** 2 / (len(y))

        seq = x
        outs = y
        seq = ToVariable(seq).to(device)
        seq = seq.view(len(seq), 1)
        outs = ToVariable(outs).to(device)
        outs = outs.view(len(outs), 1)
        # 由于输入的时array 改为 a x 1 的格式
        # 修改完之后有明显降低
        # outs = torch.from_numpy(np.array([outs]))

        # 清除网络状态
        model.zero_grad()
        # optimizer.zero_grad()
        # 重新初始化隐藏层数据
        model.hidden = model.init_hidden()
        modout = model(seq).to(device)

        loss = loss_function(modout, outs)
        # 反向传播求梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 放里头一直更新
        # statistics
        # running_loss += loss.item()
        running_loss = loss.item()
        # 临时绘图

        # 测试区段数据
        Testloss = 0
        # 测试集长度
        testdata = testdata[:1]
        testdatalen = len(testdata)
        for t1 in testdata:
            testx = ToVariable(t1[0]).view(len(t1[0]), 1).to(device)
            # model input 必须是 [xxx,1]
            with torch.no_grad():
                predDat = model(testx).data.cpu()
            predDat = predDat.numpy().reshape(-1)
            trueDat = t1[1]
            # 测试误差
            Tloss_truDat = np.array(trueDat)
            Testloss0 = loss_function(ToVariable(predDat).view(len(predDat), 1),
                                      ToVariable(Tloss_truDat).view(len(Tloss_truDat), 1))
            Testloss += Testloss0

            # 绘制最后一步的测试
            if t1 == testdata[-1]:
                vis.line(np.column_stack((Tloss_truDat, predDat)), win='trueDat',
                         opts=dict(
                             legend=['trueDat', 'predDat'],
                             title='trueDat'
                         ))

        Testloss = Testloss / testdatalen

        # 计算轨道谱
        f = np.linspace(0, 2, len(predDat))
        Pxx = 2 * abs(np.fft.fft(predDat)) ** 2 / (4 * len(predDat))
        f0 = np.linspace(0, 2, len(testdata[0][0]))
        Pxx0 = 2 * abs(np.fft.fft(testdata[0][0])) ** 2 / (4 * len(testdata[0][0]))

        vis.line(X=np.column_stack((f, f0)),
                 Y=np.column_stack((Pxx.reshape(-1), Pxx0.reshape(-1))), win='PSD',
                 opts=dict(
                     legend=['PSD-LSTM', 'TEST'],
                     title='PSD',
                     ytype='log',
                     xtype='log'
                 ))
        #        np.array(Pxx.reshape(-1), f)

        # # 简单绘图
        # # simplot(trueDat, predDat)

        rloss.append(running_loss)
        Tloss_list.append(Testloss)

        vis.line(np.column_stack((rloss, Tloss_list)), win='model Loss',
                 opts=dict(
                     legend=['Training Loss', 'Test Loss'],
                     title='Loss'
                 ))

#        print(' Epoch[{}/{}], loss:{:.6f}， Tloss:{:.6f}'.format(epoch, num_epochs,
#                                                                running_loss, Testloss))
        vis.text(' Epoch[{}/{}], loss:{:.6f}， Tloss:{:.6f}'.format(epoch, num_epochs,
                                                                   running_loss,
                                                                   Testloss),
                 win='Training Message')

        # # 计算与测试集的误差
        # predDat = model(ToVariable(x[-testlen:]).to(device)).data.cpu()
        # predDat = np.array(predDat)
        # trueDat = y[-testlen:]
        # Testloss = loss_function(modout, outs)

        # 保存模型参数
        if epoch % 200 == 0:
            pklname = 'param_N{}_layer{}_Len{}_Ep{}_St{}_CASE{}_epoch{}.pkl'.format(
                NEUNUM,
                NLAYER,
                testlen,
                num_epochs,
                step, caseN, epoch)
            # 保存误差列表
            try:
                save2excel([rloss, Tloss_list],
                           xlname='LossHistr1t20916-CASE%d.xls' % caseN)
            except:
                print('Loss saving failed.')
            # save2excel([Tloss_list], xlname='TestLossHist0912.xls')
            torch.save(model.state_dict(), pklname)
#            print("> Parameters have been saved.")

        # loss 阈值
        if running_loss < 0.01:
            print('Loss limit achived!')
            break

    # 最终测试
    predDat = model(ToVariable(x[-2 * testlen:]).to(device)).data.cpu()
    predDat = np.array(predDat)
    trueDat = y[-2 * testlen:]
    fig = plt.figure()
    plt.plot(trueDat, label='Turedata')
    plt.plot(predDat, label='Predict', alpha=0.4)
    plt.legend()
    plt.show()
    # 保存至EXCEL
    # save2excel([trueDat, predDat], xlname='Pred_Truth2.xls')
    save2excel([trueDat, predDat], xlname='Final_lstm_TestData0916-CASE%d.xls' % caseN)


if __name__ == '__main__':
    casepath = r'.\1010超参数计算.xlsx'
    workbook = xlrd.open_workbook(casepath)
    sheet2 = workbook.sheet_by_index(0)
    nrows = sheet2.nrows
    # 获取整行和整列的值（数组）

    for i in range(1, nrows):
        paramC = sheet2.row_values(i)
        try:
            print('Case ',i ,'/',nrows-1)
            main(NEUNUM = int(paramC[0]),
                 NLAYER = int(paramC[1]),
                 testlen = int(paramC[2]),
                 num_epochs = int(paramC[3]),
                 caseN = int(paramC[4]))
            torch.cuda.empty_cache()
        except:
            print('!!! Case Error')
