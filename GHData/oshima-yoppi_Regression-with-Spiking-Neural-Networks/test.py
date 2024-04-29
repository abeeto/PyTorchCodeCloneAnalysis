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
model = network.Conv4Regression(num_time=args.time,l_tau=args.tau, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
# print(args.number)
print(f'args.n:{args.number}')
model_path = f'models/{args.number}.pth'
model.load_state_dict(torch.load(model_path))


model = model.to(device)
print("building model")
print(model.state_dict().keys())
epochs = args.epoch
before_loss = None
loss_hist = []
test_hist = []
test_loss = []
th = 100
analysis_loss = [[] for _ in range(int(300*2/th))]
analysis_rate = [[] for _ in range(int(300*2/th))]



try:    
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(tqdm(test_iter, desc='test_iter')):
            # if i == 2:
            #     break
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels)
            los = loss.compute_loss(output, labels)
            test_loss.append(los.item())
            # if labels[:,0].item() // th == -6:
                # print(labels[:,0].item())
            analysis_loss[int((labels[:,0].item() + 300) / th)].append(np.sqrt(los.item()))
            analysis_rate[int((labels[:,0].item() + 300) / th)].append(abs(np.sqrt(los.item())*100/labels[:,0].item()))

except:
    traceback.print_exc()
    pass


# for i in range(len(analysis_loss)):
#     analysis_loss[i] = np.mean(analysis_loss[i])
#     analysis_rate[i] = np.mean(analysis_rate[i])
print(analysis_loss)





x = []
for i in range(int(300*2/th)):
    x.append(-300 + th/2 + th *i)



fig = plt.figure(f'{model_path}のloss分析')
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# ax1.plot(x, analysis_loss)
ax1.boxplot(analysis_loss, labels=x)
ax1.set_xlabel('Angular Velocity')
ax1.set_ylabel('loss')
ax2.boxplot(analysis_rate, labels=x)
ax2.set_xlabel('Angular Velocity')
ax2.set_ylabel('loss rate[%]')
plt.tight_layout()
plt.show()
 



