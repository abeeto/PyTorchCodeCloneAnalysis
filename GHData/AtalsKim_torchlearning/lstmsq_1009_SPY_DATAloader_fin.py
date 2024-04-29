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
import torch.utils.data as Data

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

    def __init__(self, input_size, hidden_dim, batchsize=1, num_layer=1, out_size=1):
        super(LSTMpred, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.batchsize = batchsize
        self.hidden = self.init_hidden()
        # 数据归一化操作
        # self.bn1 = nn.BatchNorm1d(num_features=320)
        # 增加DROPout 避免过拟合
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layer, dropout=0.2)
        # outfeature = 1
        self.hidden2out = nn.Linear(self.hidden_dim, out_size)

    # 第一个求导应该不用的吧
    def init_hidden(self):
        # print('hidden to zeros')
        return (
            Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).to(
                device),
            Variable(torch.zeros(self.num_layer, self.batchsize, self.hidden_dim)).to(
                device))

    def forward(self, seq, h_state):
        # 三个句子，10个单词，1000

        # hc维度应该是 [层数，batch, hiddensize]
        # out 维度应该是[单词, batch, hiddensize]
        # lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        # seq =1  batch 1 vec 200

        # vecinput 行数据的个数
        # seq = 1组里头有多少数据，batchsize 喂多少组， input维数
        # input >>> [seq_len, batchsize, input_size]
        # out >>> [seq_len, bathchsize, hiddenlayernum]
        # h,c >>> [层数，batchsize, hiddensize]
        # lstm_out, self.hidden = self.lstm(
        #     seq.view(-1, self.batchsize, self.input_size), self.hidden)
        # 是不是多对一的话留下最后结果
        # outdat = self.hidden2out(lstm_out[-1].view(self.batchsize, -1))
        # return outdat.view(-1)
        lstm_out, h_state = self.lstm(
            seq.view(-1, self.batchsize, self.input_size), h_state)
        # 改成线性层输出形式
        s, b, h = lstm_out.shape
        x = lstm_out.view(s * b, h)
        outdat = self.hidden2out(x)
        # outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat, h_state

        # # 莫凡的方法，循环和第一种结果一样
        # mf_out = []
        # for time_s in range(lstm_out.size(0)):
        #     mf_out.append(self.hidden2out(lstm_out[time_s,:,:]))
        # mf_out = torch.stack(mf_out, dim=1)
        # # 变成 N X 1
        # mf_out = mf_out.view(-1,1)


def main(NEUNUM=16, NLAYER=4, testlen=1000, num_epochs=10000, caseN=9999):
    TESTMOD = False
    LOADPKL = False
    # batchsize
    step = 4
    port = 6007
    loadlen = 80000
    testdatlen = 2000

    visdomenv = 'PytorchTest%d' % caseN
    # inputsize 这个应该是指特征的维度，所以是1

    # 模型构建
    if LOADPKL:
        model = load_net_state(1, 1)
    else:
        #
        minibatch = 1
        model = LSTMpred(1, NEUNUM, minibatch, NLAYER, out_size=1).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.01)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_function = nn.MSELoss()
    mparanum = print_model_parm_nums(model)

    # # 数据读取
    xlpath = r'Prep_BJ_GJC_W2014010602.xlsx'
    # 测试
    # xlpath = r'Prep_BJ_GJC_W2014010602sine.xlsx'

    ## 训练集
    x, y = loaddata(xlpath, length=loadlen, start=1)
    # 使用dataloader
    torch_dataset = Data.TensorDataset(ToVariable(x), ToVariable(y))
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=testlen,
                             shuffle=False,
                             drop_last=True)
    # 原始
    trainlen = round(len(x) * 0.8)
    # dat = trainDataGen(x[:trainlen], y[:trainlen], testlen, step=step)
    # testdata = trainDataGen(x[trainlen:], y[trainlen:], testlen, step=step)

    ## 测试集
    x_t, y_t = loaddata(xlpath, length=testdatlen, start=loadlen + 1)
    # 使用dataloader
    test_dataset = Data.TensorDataset(ToVariable(x_t), ToVariable(y_t))
    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=testlen,
                                  shuffle=False,
                                  drop_last=True)

    print('Training Loader Num:' ,len(loader) ,'Testing Loader Num:',len(test_loader))

    # LOSS 计算
    rloss = []
    Tloss_list = []
    print(
        "NNumber:{}, TestLen:{}, Epochs:{}, Nlayer:{}, paranum:{}".format(NEUNUM, testlen,
                                                                          num_epochs,
                                                                          NLAYER,
                                                                          mparanum))

    running_loss = 0.0
    # 使用VISDOM 进行绘图
    vis = visdom.Visdom(env=visdomenv, port=port)


    # 开始训练
    #    _ = input('Press any key to continue.......')
    print("Epoch Start...")

    for epoch in range(num_epochs):

        eval_loss = 0
        for seq, outs in loader:
            seq = Variable(seq).view(-1, 1).to(device)
            outs = Variable(outs).view(-1, 1).to(device)

            # 清除网络状态
            optimizer.zero_grad()
            # 重新初始化隐藏层数据
            h_state = model.init_hidden()

            modout, h_state = model(seq, h_state)
            modout = modout.to(device)
            loss = loss_function(modout, outs)
            # loss.requires_grad = True
            # requires_grad = True

            # 绘制train 图
            vis.line(np.column_stack((outs.view(-1).cpu().detach().numpy(),
                                      modout.view(-1).cpu().detach().numpy())),
                     win='Trainamp',
                     opts=dict(
                         legend=['trueDat', 'predDat'],
                         title='Training Verification'
                     ))

            # 反向传播求梯度
            loss.backward()
            # loss.backward()
            # 更新梯度
            optimizer.step()
            eval_loss += loss.item()

        running_loss = eval_loss / len(loader)

        # 测试区段LOSS
        Testloss = 0
        # 测试集长度
        testset_num = len(test_loader)
        for inp_t, outs_t in test_loader:
            inp_t = inp_t.view(-1, 1).to(device)
            outs_t = outs_t.view(-1, 1).cpu()
            with torch.no_grad():
                h_state = model.init_hidden()
                predDat, _ = model(inp_t, h_state)
                predDat = predDat.cpu()
                # 测试绘图
                vis.line(np.column_stack((outs_t, predDat)), win='trueDat',
                         opts=dict(
                             legend=['trueDat', 'predDat'],
                             title='Testing Verification'
                         ))

            # 测试误差
            Testloss0 = loss_function(outs_t, predDat).item()
            Testloss += Testloss0


        Testloss = Testloss / testset_num

        # 计算轨道谱，就拿最后的算
        f = np.linspace(0, 2, len(predDat))
        Pxx = 2 * abs(np.fft.fft(predDat)) ** 2 / (4 * len(predDat))
        f0 = np.linspace(0, 2, len(outs_t))
        Pxx0 = 2 * abs(np.fft.fft(outs_t)) ** 2 / (4 * len(outs_t))
        vis.line(X=np.column_stack((f, f0)),
                 Y=np.column_stack((Pxx.reshape(-1), Pxx0.reshape(-1))), win='PSD',
                 opts=dict(
                     legend=['PSD-LSTM', 'TEST'],
                     title='PSD',
                     ytype='log',
                     xtype='log'
                 ))

        rloss.append(running_loss)
        Tloss_list.append(Testloss)
        vis.line(np.column_stack((rloss, Tloss_list)), win='model Loss',
                 opts=dict(
                     legend=['Training Loss', 'Test Loss'],
                     title='Loss'
                 ))
        # 输出epoch信息
        print('\rEpoch[{}/{}], loss:{:.6f}， Tloss:{:.6f}'.format(epoch, num_epochs,
                                                                  running_loss, Testloss),
              end='')

        # 输出epoch信息，visdom 模型信息
        vis.text('NNumber:{}\nTestLen:{}\nEpochs:{}\nNlayer:{}\nparanum:{}\nEpoch[{}/{}]\nloss:{:.6f}\nTloss:{:.6f}'\
                 .format(NEUNUM,testlen,num_epochs,NLAYER,mparanum,epoch,num_epochs,running_loss,Testloss),
                 win='Training Message')

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

    # # 最终测试，对比部分可以删除
    # predDat = model(ToVariable(x[-2 * testlen:]).to(device)).data.cpu()
    # predDat = np.array(predDat)
    # trueDat = y[-2 * testlen:]
    # fig = plt.figure()
    # plt.plot(trueDat, label='Turedata')
    # plt.plot(predDat, label='Predict', alpha=0.4)
    # plt.legend()
    # plt.show()

    ## 保存至EXCEL
    # save2excel([trueDat, predDat], xlname='Pred_Truth2.xls')
    # save2excel([trueDat, predDat], xlname='Final_lstm_TestData0916-CASE%d.xls' % caseN)


if __name__ == '__main__':
    casepath = r'.\1010超参数计算.xlsx'
    workbook = xlrd.open_workbook(casepath)
    sheet2 = workbook.sheet_by_index(0)
    nrows = sheet2.nrows
    # 获取整行和整列的值（数组）

    for i in range(1, nrows):
        paramC = sheet2.row_values(i)
        try:
            print('Case ', i, '/', nrows - 1, '\n Name: ', paramC[4])
            main(NEUNUM=int(paramC[0]),
                 NLAYER=int(paramC[1]),
                 testlen=int(paramC[2]),
                 num_epochs=int(paramC[3]),
                 caseN=int(paramC[4]))
            torch.cuda.empty_cache()
        except Exception as err:
            print('!!! Case Error')
            print(err)
