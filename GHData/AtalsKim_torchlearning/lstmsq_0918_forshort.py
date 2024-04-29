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






#






# 均值、方差归一化
def MS_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std, mu, std

def MS_Fnormalize(data, mu, std):
    return data * std + mu

# 单维最大最小归一化
def MMNormalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i] - low) / delta
    return list, low, high


# 反归一化函数
def MMFNoramlize(list, low, high):
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
    print("Excel out finished.")
    print(os.path.abspath(xlname))
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
        inputs = sheet2.col_values(9, start_rowx=start)
        targets = sheet2.col_values(11, start_rowx=start)
    else:
        inputs = sheet2.col_values(9, start_rowx=start, end_rowx=start + length)
        targets = sheet2.col_values(11, start_rowx=start, end_rowx=start + length)
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


def main():
    TESTMOD = False
    NEUNUM = 128
    NLAYER = 2
    # batchsize
    testlen = 3000
    step = 4
    num_epochs = 10000
    port = 6007
    # os.popen(r"python -m visdom.server -port %d"%port)

    # inputsize 这个应该是指特征的维度，所以是1
    model = LSTMpred(1, NEUNUM, testlen, NLAYER).to(device)
    # 改用Adam
    # L2正则化 weight_decay = 0.01
    optimizerSGD = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.01)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    loss_function = nn.MSELoss()
    # 配置输入batch
    # xlpath = r'excelTest37000.xlsx'
    xlpath = r'Prep_DATA0915.xlsx'
    # xlpath = r'SINETEST1000.xls'
    # 12500~ 15000 有坏值
    x, y = loaddata(xlpath, length=90000, start=1)

    trainlen = round(len(x) * 0.8)
    # 后1000 留下测试

    # 数据划分
    traindatx = [x[:trainlen], y[:trainlen]]
    testdata = [x[trainlen:], y[trainlen:]]


# 不用原始的batch
    # dat = trainDataGen(x[:trainlen], y[:trainlen], testlen, step=step)
    # # 测试集
    # testdata = trainDataGen(x[trainlen:], y[trainlen:], testlen, step=step)


    # 模型计算总参数
    mparanum = print_model_parm_nums(model)
    rloss = []
    Tloss_list = []
    print(
        "NNumber:{}, TestLen:{}, Epochs:{}, Nlayer:{}, paranum:{}".format(NEUNUM, testlen,
                                                                          num_epochs,
                                                                          NLAYER,
                                                                          mparanum))

    # ship_train_loader = DataLoader(ship_train_dataset, batch_size=16, num_workers=4,
    #                                shuffle=False, **kwargs)

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

    print("Epoch Start...")

    running_loss = 0.0

    # 使用VISDOM 进行绘图
    vis = visdom.Visdom(env='PytorchTest', port=port)
    vis.text(
        "NNumber:{}, TestLen:{}, Epochs:{}, Nlayer:{}, paranum:{}".format(NEUNUM, testlen,
                                                                          num_epochs,
                                                                          NLAYER,
                                                                          mparanum),
        win='Training set')
    seq = traindatx[0]
    outs = traindatx[1]
    # 归一化处理
    seq0, tmu1, tstd1 = MS_normalize(np.array(seq))
    outs0, tmu2, tstd2 = MS_normalize(np.array(outs))
    for epoch in range(num_epochs):
        # print(epoch, end='')

        seq = ToVariable(seq0).to(device)
        seq = seq.view(len(seq), 1)
        outs = ToVariable(outs0).to(device)
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
        # 整体测试


        t1 = testdata
        # 测试数据的输入归一化
        t1regu, _, _ = MS_normalize(t1[0])
        testx = ToVariable(t1regu).view(len(t1regu), 1).to(device)
        # model input 必须是 [xxx,1],d
        with torch.no_grad():
            predDat = model(testx).data.cpu()
        predDat = predDat.numpy().reshape(-1)

        # 反归一化处理
        predDat = MS_Fnormalize(predDat, tmu2, tmu2)

        trueDat = t1[1]
        # 测试误差
        Tloss_truDat = np.array(trueDat)
        Testloss0 = loss_function(ToVariable(predDat).view(len(predDat), 1),
                                  ToVariable(Tloss_truDat).view(len(Tloss_truDat), 1))
        Testloss += Testloss0
        # 绘制最后一步的测试
        vis.line(Tloss_truDat, win='trueDat',
                 opts=dict(
                     legend=['trueDat'],
                     title='trueDat'
                 ))
        vis.line(predDat, win='predDat',
                 opts=dict(
                     legend=['predDat'],
                     title='predDat'
                 ))

        # # 简单绘图
        # # simplot(trueDat, predDat)

        rloss.append(running_loss)
        Tloss_list.append(Testloss)

        vis.line(rloss, win='model Loss',
                 opts=dict(
                     legend=['Training Loss'],
                     title='Training Loss'
                 ))
        vis.line(Tloss_list, win='Test Loss',
                 opts=dict(
                     legend=['Test Loss'],
                     title='Test Loss'
                 ))

        print(' Epoch[{}/{}], loss:{:.6f}， Tloss:{:.6f}'.format(epoch, num_epochs,
                                                                running_loss, Testloss))
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
        # 保存模型
        if epoch % 50 == 0:
            pklname = 'param_N{}_layer{}_Len{}_Ep{}_St{}0916change.pkl'.format(NEUNUM,
                                                                step)
            # 保存误差列表
            try:
                save2excel([rloss, Tloss_list], xlname='LossHistr1t20916.xls')
            except:
                print('Loss saving failed.')
            # save2excel([Tloss_list], xlname='TestLossHist0912.xls')
            torch.save(model.state_dict(), pklname)
            print("> Parameters have been saved.")

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
    save2excel([trueDat, predDat], xlname='Final_lstm_TestData0916.xls')


if __name__ == '__main__':
    main()