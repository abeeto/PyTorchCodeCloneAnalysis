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
import traceback
import pandas as pd
from model import snu_layer
from model import network
from model import loss
from tqdm import tqdm
# from torchsummary import summary
import argparse
import pickle
import time

def analyze(model, device, test_iter, loss_hist=[], test_hist=[],
            start_time=0, end_time=10, epoch=0, lr=None, tau=None ):
    """"
    analyze trained model 
    """
    model = model.to(device)
    # print("building model")
    # print(model.state_dict().keys())
    # epochs = args.epoch
    # before_loss = None
    # loss_hist = []
    # test_hist = []
    test_loss = []
    
    # エラーの解析
    # 何度ずつ区切るか
    th = 4
    analysis_loss = [[] for _ in range(int(20*2/th))]
    analysis_rate = [[] for _ in range(int(20*2/th))]

    # 統計的な解析用
    loss_ = []
    rate_ = []
    distribution_loss = [0]*40
    distribution_rate = [0]*200
    test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "test" ,time = 20)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=True)
    try:    
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(tqdm(test_iter, desc='test_iter')):
                # if i == 2:
                #     break
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                los = loss.compute_loss(output, labels)
                test_loss.append(los.item())


                analysis_loss[int((labels[:,0].item() + 20) / th)].append(np.sqrt(los.item()))
                analysis_rate[int((labels[:,0].item() + 20) / th)].append(abs(np.sqrt(los.item())*100/labels[:,0].item()))

                loss_.append(los.item())
                rate_.append(abs(np.sqrt(los.item())*100/labels[:,0].item()))
                try:
                    distribution_loss[int(np.sqrt(los.item()))] += 1
                except:
                    distribution_loss[-1] += 1
                try:
                    distribution_rate[int(abs(np.sqrt(los.item())*100/labels[:,0].item()))] += 1
                except:
                    distribution_loss[-1] += 1

    except:
        traceback.print_exc()
        pass


  
    # print(analysis_loss)
    





    x = []
    for i in range(int(20*2/th)):
        x.append(-20 + th/2 + th *i)
    
    ana_x = x
    def sqrt_(n):
        return n ** 0.5
    ###ログのグラフ

    ax1_x = []
    for i in range(len(loss_hist)):
        ax1_x.append(i+1)
    ax2_x = []
    for i in range(len(test_hist)):
        ax2_x.append(i + 1)
    epoch += 0.0001
    time_ = (end_time - start_time)/(3600*epoch)
    time_ = '{:.2f}'.format(time_)
    fig = plt.figure(f'学習時間:{time_}h/epoch, τ:{tau}, 学習率:{lr}', figsize=(8,12))
    ax1 = fig.add_subplot(4, 2, 1)
    ax2 = fig.add_subplot(4, 2, 2)
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)
    ax5 = fig.add_subplot(4, 2, 5)
    ax6 = fig.add_subplot(4, 2, 6)
    ax7 = fig.add_subplot(4, 2, 7)
    ax8 = fig.add_subplot(4, 2, 8)


    loss_hist = list(map(sqrt_, loss_hist))
    test_hist = list(map(sqrt_, test_hist))
    ax1.plot(ax1_x, loss_hist)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss_hist')
    ax2.plot(ax2_x, test_hist)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('test_hist')

    
    ax3.boxplot(analysis_loss, showmeans=True)
    ax3.set_xlabel('Angular Velocity')
    ax3.set_ylabel('Loss')
    # ax3.set_ylim(0, 40)
    ax4.boxplot(analysis_rate, showmeans=True)
    ax4.set_xlabel('Angular Velocity')
    ax4.set_ylabel('Loss Rate[%]')
    # ax4.set_ylim(0, 200)


    ax5.boxplot(analysis_loss, showmeans=True)
    ax5.set_xlabel('Angular Velocity (unedited)')
    ax5.set_ylabel('Loss')
    ax6.boxplot(analysis_rate, showmeans=True)
    ax6.set_xlabel('Angular Velocity (unedited)')
    ax6.set_ylabel('Loss Rate[%]')

    std_loss = np.std(loss_)
    std_rate = np.std(rate_)
    mean_loss = np.mean(loss_)
    mean_rate = np.mean(rate_)
    ax7.plot(distribution_loss)
    ax7.set_xlabel(f'Loss | mean:{round(mean_loss, 1)}, std:{round(std_loss,1)}')
    ax7.set_ylabel('Count')
    ax8.plot(distribution_rate)
    ax8.set_xlabel(f'Loss Rate[%] | mean:{round(mean_rate,1)}, std:{round(std_rate, 1)}')
    ax8.set_ylabel('Count')


    plt.tight_layout()
    plt.show()
    
    return x, analysis_loss, analysis_rate

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



def save(string, what):
    save_path = f"analysis/{string}.pickle"
    with open(save_path, mode='wb') as f:
        pickle.dump(what, f)
    return 
def save_loss(model, device, save_path='../analysis/loss.csv'):
    model = model.to(device)
    
    test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output_vector/', which = "test" ,time = 20)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=True)
    df_loss = {}
    lst = ['label_x', 'label_y','label_z','label_w', 'loss_x', 'loss_y', 'loss_z', 'loss_w']
    for i in lst:
        df_loss[i] = []
    try:    
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(tqdm(test_iter, desc='test_iter')):
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                loss_x, loss_y, loss_z, loss_omega, same_loss = loss.analysis_loss(output, labels)
                label_x = labels[:,0].item()
                label_y = labels[:,1].item()
                label_z = labels[:,2].item()
                loss_x = loss_x.item()
                loss_y = loss_y.item()
                loss_z = loss_z.item()
                loss_w = loss_omega.item()
                label_w = (label_x**2 + label_y**2 + label_z**2)** 0.5
                df_loss ['label_x'].append(label_x)
                df_loss ['label_y'].append(label_y)
                df_loss ['label_z'].append(label_z)
                df_loss ['label_w'].append(label_w)
                df_loss ['loss_x'].append(loss_x)
                df_loss ['loss_y'].append(loss_y)
                df_loss ['loss_z'].append(loss_z)
                df_loss ['loss_w'].append(loss_w)
    except:
        traceback.print_exc()
        pass
    df = pd.DataFrame(df_loss)
    df.to_csv(save_path, index=False)
    return df
    
    
    
def analyze_3vector(model, device, test_iter, loss_hist=[], test_hist=[],
            start_time=0, end_time=10, epoch=0, lr=None, tau=None ):
    """"
    analyze trained model 
    """
    model = model.to(device)
    test_loss = []
    
    # エラーの解析
    # 何度ずつ区切るか
    th = 5
    # 統計的な解析用
    loss_ = []
    rate_ = []
    distribution_loss = [0]*40
    distribution_rate = [0]*200
    test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output_vector/', which = "test" ,time = 20)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=True)

    analysis_x = AnalyzeDataset(th=th)
    analysis_y = AnalyzeDataset(th=th)
    analysis_z = AnalyzeDataset(th=th)
    analysis_w = AnalyzeDataset(th=th)
    try:    
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(tqdm(test_iter, desc='test_iter')):
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                los=None
                loss_x, loss_y, loss_z, loss_omega, same_loss = loss.analysis_loss(output, labels)
                label_x = labels[:,0].item()
                label_y = labels[:,1].item()
                label_z = labels[:,2].item()
                loss_x = loss_x.item()
                loss_y = loss_y.item()
                loss_z = loss_z.item()
                loss_w = loss_omega.item()
                label_w = (label_x**2 + label_y**2 + label_z**2)** 0.5
                test_loss.append(same_loss.item())


                # analysis_loss[int((labels[:,0].item() + 20) / th)].append(np.sqrt(los.item()))
                analysis_x.add(loss=loss_x, label=label_x)
                analysis_y.add(loss=loss_y, label=label_y)
                analysis_z.add(loss=loss_z, label=label_z)
                analysis_w.add(loss=loss_w, label=label_w)
                # loss_.append(los.item())
                # rate_.append(abs(np.sqrt(los.item())*100/labels[:,0].item()))
                # try:
                #     distribution_loss[int(np.sqrt(los.item()))] += 1
                # except:
                #     distribution_loss[-1] += 1
                # try:
                #     distribution_rate[int(abs(np.sqrt(los.item())*100/labels[:,0].item()))] += 1
                # except:
                #     distribution_loss[-1] += 1

    except:
        traceback.print_exc()
        pass
    
    
    
    
    save('x', analysis_x)
    save('y', analysis_y)
    save('z', analysis_z)
    save('w', analysis_w)




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
    return





if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--epoch', '-e', type=int, default=10)##英さんはepoc100だった
    parser.add_argument('--time', '-t', type=int, default=20,
                            help='Total simulation time steps.')
    parser.add_argument('--rec', '-r', action='store_true' ,default=False)  # -r付けるとTrue                  
    parser.add_argument('--forget', '-f', action='store_true' ,default=False) 
    parser.add_argument('--dual', '-d', action='store_true' ,default=False)
    parser.add_argument('--number', '-n', type=int)
    parser.add_argument('--tau', type=float)
    args = parser.parse_args()


    print("***************************")
    train_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "train" ,time = args.time)
    test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "test" ,time = args.time)
    data_id = 2
    # print(train_dataset[data_id][0]) #(784, 100) 
    train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)
    # print(train_iter.shape)
    # ネットワーク設計
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 畳み込みオートエンコーダー　リカレントSNN　
    # model = network.SNU_Regression(num_time=args.time,l_tau=0.8, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
    # model = network.Conv4Regression(num_time=args.time,l_tau=args.tau, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
    model = network.VectorRegression(num_time=args.time,l_tau=args.tau, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
    # print(args.number)
    print(f'args.n:{args.number}')
    model_path = f'models/{args.number}.pth'
    model.load_state_dict(torch.load(model_path))
    save_loss(model, device=device, save_path='analysis/loss.csv')
    # analyze_3vector(model, device=device, test_iter=test_iter, tau=args.tau)
    

