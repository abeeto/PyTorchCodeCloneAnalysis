import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from data import LoadDataset
import os 
from tqdm import tqdm
import datetime
# from rectangle_builder import rectangle,test_img
import traceback

from model import snu_layer
from model import network
from model import loss
from tqdm import tqdm
# from torchsummary import summary
import argparse
import time
import pickle
import seaborn as sns
import pandas as pd



class AnalyzeDataset:
    def __init__(self, th=4, max_angle=20,):
        # 何度ずつ区切るか
        self.th = th
        self.max = max_angle
        #　plt用の軸を用意。
        self.axis = []
        for i in range(int(2*self.max/self.th)):
            low = self.th * i - self.max
            high = low + self.th
            moji = f'[{low},{high})'
            self.axis.append(moji)

        self.loss = [[] for _ in range(int(self.max*2/self.th))]
        self.loss_rate = [[] for _ in range(int(self.max*2/self.th))]
    def add(self, loss, label):
        self.loss[int((label + self.max) / self.th)].append(loss)
        self.loss_rate[int((label + self.max) / self.th)].append(abs(loss/label)*100)
        return



def read(string):
    load_path = f"analysis/{string}.pickle"
    with open(load_path, mode='rb') as f:
        data = pickle.load(f)
    return data

def make_df(dataset):
    num = []
    axis = []
    value = []
    length = len(dataset[0])
    for axis_ ,data in enumerate(dataset):
        for i in range(length):
            for j in data[i]:
                num.append(i)
                value.append(j)
                #　asciiコードを用いてx,y,zに変換
                moji = "w_"
                moji += chr(120+axis_)
                axis.append(moji)
    df = pd.DataFrame({
        'number' : num,
        'axis' : axis,
        'value' : value 
    })
    return df
    




    return 

def analyze_read(loss_hist=[], test_hist=[],
            start_time=0, end_time=10, epoch=0, lr=None, tau=None ):
    """"
    .pickleで保存されたデータを用いて分析する関数
    
    """
    test_loss = []
    
    # エラーの解析
    # 何度ずつ区切るか
    th = 5
    # 統計的な解析用
    loss_ = []
    rate_ = []
    distribution_loss = [0]*40
    distribution_rate = [0]*200


    # 保存された分析結果の読み込み
    analysis_x = read('x')
    analysis_y = read('y')
    analysis_z = read('z')
    analysis_w = read('w')
    # print(len(analysis_x.loss))

  




    ax1_x = []
    for i in range(len(loss_hist)):
        ax1_x.append(i+1)
    ax2_x = []
    for i in range(len(test_hist)):
        ax2_x.append(i + 1)
    epoch += 0.0001
    time_ = (end_time - start_time)/(3600*epoch)
    time_ = '{:.2f}'.format(time_)
    # fig = plt.figure(f'学習時間:{time_}h/epoch, τ:{tau}, 学習率:{lr}', figsize=(18, 9))
    fig = plt.figure(f'学習時間:{time_}h/epoch, τ:{tau}, 学習率:{lr}', figsize=(18, 9))
    ax1 = fig.add_subplot(4, 3, 1)
    ax2 = fig.add_subplot(4, 3, 2)
    # ax3 = fig.add_subplot(4, 3, 3)
    ax4 = fig.add_subplot(4, 3, 4)
    ax5 = fig.add_subplot(4, 3, 5)
    ax6 = fig.add_subplot(4, 3, 6)
    ax7 = fig.add_subplot(4, 3, 7)
    ax8 = fig.add_subplot(4, 3, 8)
    ax9 = fig.add_subplot(4, 3, 9)
    ax10 = fig.add_subplot(4, 3, 10)
    ax11 = fig.add_subplot(4, 3, 11)
    # ax12 = fig.add_subplot(4, 3, 12)
    



    ax1.plot(ax1_x, loss_hist)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss_hist')
    ax2.plot(ax2_x, test_hist)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('test_hist')

    
    ax4.boxplot(analysis_x.loss,  showmeans=True)
    ax4.set_xlabel('w_x')
    ax4.set_xticklabels(analysis_x.axis)
    ax4.set_ylabel('Error')
    
    ax5.boxplot(analysis_y.loss, showmeans=True)
    ax5.set_xlabel('w_y')
    ax5.set_ylabel('Error')
    ax5.set_xticklabels(analysis_y.axis)
    
    ax6.boxplot(analysis_z.loss, showmeans=True)
    ax6.set_xlabel('w_z')
    ax6.set_ylabel('Error')
    ax6.set_xticklabels(analysis_z.axis)

    

    
    ax7.boxplot(analysis_x.loss_rate, showmeans=True)
    ax7.set_xlabel('w_x')
    ax7.set_ylabel('Error Rate[%]')
    ax7.set_xticklabels(analysis_x.axis)

    ax8.boxplot(analysis_y.loss_rate, showmeans=True)
    ax8.set_xlabel('w_y')
    ax8.set_ylabel('Error Rate[%]')
    ax8.set_xticklabels(analysis_y.axis)

    ax9.boxplot(analysis_z.loss_rate, showmeans=True)
    ax9.set_xlabel('w_z')
    ax9.set_ylabel('Error Rate[%]')
    ax9.set_xticklabels(analysis_z.axis)

    ax10.boxplot(analysis_w.loss, showmeans=True)
    ax10.set_xlabel('Angular Velocity w')
    ax10.set_ylabel('Error')
    ax10.set_xticklabels(analysis_w.axis)

    ax11.boxplot(analysis_w.loss_rate, showmeans=True)
    ax11.set_xlabel('Angular Velocity w')
    ax11.set_ylabel('Error Rate[%]')
    ax11.set_xticklabels(analysis_w.axis)



    plt.tight_layout()
    plt.show()
    fig = plt.figure(f'学習時間:{time_}h/epoch, τ:{tau}, 学習率:{lr}', figsize=(18, 9))
    ax1 = fig.add_subplot(4, 3, 1)
    ax2 = fig.add_subplot(4, 3, 2)
    # ax3 = fig.add_subplot(4, 3, 3)
    ax4 = fig.add_subplot(4, 3, 4)
    ax5 = fig.add_subplot(4, 3, 5)
    ax6 = fig.add_subplot(4, 3, 6)
    ax7 = fig.add_subplot(4, 3, 7)
    ax8 = fig.add_subplot(4, 3, 8)
    ax9 = fig.add_subplot(4, 3, 9)
    ax10 = fig.add_subplot(4, 3, 10)
    ax11 = fig.add_subplot(4, 3, 11)
    # ax12 = fig.add_subplot(4, 3, 12)



    ax1.plot(ax1_x, loss_hist)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss_hist')
    ax2.plot(ax2_x, test_hist)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('test_hist')

    
    ax4.boxplot(analysis_x.loss,  showmeans=False, sym="")
    ax4.set_xlabel('w_x')
    ax4.set_xticklabels(analysis_x.axis)
    ax4.set_ylabel('Error')
    
    ax5.boxplot(analysis_y.loss, showmeans=False, sym="")
    ax5.set_xlabel('w_y')
    ax5.set_ylabel('Error')
    ax5.set_xticklabels(analysis_y.axis)
    
    ax6.boxplot(analysis_z.loss, showmeans=False, sym="")
    ax6.set_xlabel('w_z')
    ax6.set_ylabel('Error')
    ax6.set_xticklabels(analysis_z.axis)

    

    
    ax7.boxplot(analysis_x.loss_rate, showmeans=False, sym="")
    ax7.set_xlabel('w_x')
    ax7.set_ylabel('Error Rate[%]')
    ax7.set_xticklabels(analysis_x.axis)

    ax8.boxplot(analysis_y.loss_rate, showmeans=False, sym="")
    ax8.set_xlabel('w_y')
    ax8.set_ylabel('Error Rate[%]')
    ax8.set_xticklabels(analysis_y.axis)

    ax9.boxplot(analysis_z.loss_rate, showmeans=False, sym="")
    ax9.set_xlabel('w_z')
    ax9.set_ylabel('Error Rate[%]')
    ax9.set_xticklabels(analysis_z.axis)

    ax10.boxplot(analysis_w.loss, showmeans=False, sym="")
    ax10.set_xlabel('Angular Velocity w')
    ax10.set_ylabel('Error')
    ax10.set_xticklabels(analysis_w.axis)

    ax11.boxplot(analysis_w.loss_rate, showmeans=False, sym="")
    ax11.set_xlabel('Angular Velocity w')
    ax11.set_ylabel('Error Rate[%]')
    ax11.set_xticklabels(analysis_w.axis)


    



    
    plt.tight_layout()
    plt.show()
    fig = plt.figure(f'学習時間:{time_}h/epoch, τ:{tau}, 学習率:{lr}', figsize=(18, 9))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)



    ax1.plot(ax1_x, loss_hist)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss_hist')
    ax2.plot(ax2_x, test_hist)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('test_hist')

    df_error = make_df([analysis_x.loss, analysis_y.loss, analysis_z.loss])
    df_rate = make_df([analysis_x.loss_rate, analysis_y.loss_rate, analysis_z.loss_rate])
    sns.boxenplot(
        x = df_error['number'],
        y = df_error['value'],
        hue=df_error['axis'],
        ax=ax3,
        showfliers=False,
        # sym=""
    )
    ax3.set_xticklabels(analysis_x.axis)
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Error')
    sns.boxenplot(
        x = df_rate['number'],
        y = df_rate['value'],
        hue=df_rate['axis'],
        ax=ax5,
        showfliers=False
        # whis='range'
        # sym=""
    )
    ax5.set_xticklabels(analysis_x.axis)
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Error Rate[%]')
    ax5.set_ylim(0, 150)

    ax4.boxplot(analysis_w.loss, showmeans=False, sym="")
    ax4.set_xlabel('Angular Velocity w')
    ax4.set_ylabel('Error')
    ax4.set_xticklabels(analysis_w.axis)

    ax6.boxplot(analysis_w.loss_rate, showmeans=False, sym="")
    ax6.set_xlabel('Angular Velocity w')
    ax6.set_ylabel('Error Rate[%]')
    ax6.set_xlim(4.1,9)
    ax6.set_xticklabels(analysis_w.axis)


    # ax4.boxplot(analysis_x.loss,  showmeans=False, sym="")
    # ax4.set_xlabel('w_x')
    # ax4.set_xticklabels(analysis_x.axis)
    # ax4.set_ylabel('Error')
    
    # ax5.boxplot(analysis_y.loss, showmeans=False, sym="")
    # ax5.set_xlabel('w_y')
    # ax5.set_ylabel('Error')
    # ax5.set_xticklabels(analysis_y.axis)
    
    # ax6.boxplot(analysis_z.loss, showmeans=False, sym="")
    # ax6.set_xlabel('w_z')
    # ax6.set_ylabel('Error')
    # ax6.set_xticklabels(analysis_z.axis)

    

    
    # ax7.boxplot(analysis_x.loss_rate, showmeans=False, sym="")
    # ax7.set_xlabel('w_x')
    # ax7.set_ylabel('Error Rate[%]')
    # ax7.set_xticklabels(analysis_x.axis)

    # ax8.boxplot(analysis_y.loss_rate, showmeans=False, sym="")
    # ax8.set_xlabel('w_y')
    # ax8.set_ylabel('Error Rate[%]')
    # ax8.set_xticklabels(analysis_y.axis)

    # ax9.boxplot(analysis_z.loss_rate, showmeans=False, sym="")
    # ax9.set_xlabel('w_z')
    # ax9.set_ylabel('Error Rate[%]')
    # ax9.set_xticklabels(analysis_z.axis)




    



    
    plt.tight_layout()
    plt.show()
    return





if __name__ == "__main__":

    analyze_read()
    

