from statistics import mode
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
from analysis import analyze
#from mp4_rec import record, rectangle_record
import pandas as pd
# import scipy.io
# from torchsummary import summary
import argparse
import time

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=8)
parser.add_argument('--epoch', '-e', type=int, default=15)
parser.add_argument('--time', '-t', type=int, default=20,
                        help='Total simulation time steps.')
parser.add_argument('--rec', '-r', action='store_true' ,default=False)  # -r付けるとTrue                  
parser.add_argument('--forget', '-f', action='store_true' ,default=False) 
parser.add_argument('--dual', '-d', action='store_true' ,default=False)
parser.add_argument('--tau',  type=float ,default=0.8)
args = parser.parse_args()


print("***************************")
train_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output_vector/', which = "train" ,time = args.time)
test_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output_vector/', which = "test" ,time = args.time)
data_id = 2
# print(train_dataset[data_id][0]) #(784, 100) 
train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)
# print(train_iter.shape)
# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 畳み込みオートエンコーダー　リカレントSNN　
model = network.VectorRegression(num_time=args.time,l_tau=args.tau, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
# model = network.Conv4Regression(num_time=args.time,l_tau=args.tau, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch)
# model = network.RSNU(num_time=args.time,l_tau=0.8, soft =False, rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch, bias=True)



model = model.to(device)
print("building model")
print(model.state_dict().keys())
# lr = 1e-4
lr = 4e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = args.epoch
before_loss = None
loss_hist = []
test_hist = []
try:

    for epoch in tqdm(range(epochs), desc='epoch',):
        running_loss = 0.0
        local_loss = []
        test_loss = []
        print("EPOCH",epoch)
        
        # print(f'train_iter len{len(train_iter)}')
        print(f'before_loss:{before_loss}') ## 一個前のepoch loss
        for i ,(inputs, labels) in enumerate(tqdm(train_iter, desc='train')):
            optimizer.zero_grad()
            inputs = inputs[:,:args.time]
            inputs = inputs.to(device)
            labels = labels.to(device)
            torch.cuda.memory_summary(device=None, abbreviated=False)
            output= model(inputs)
            los = loss.vector_loss(output, labels)
            
            # print(output)
            # print(f'label:{labels[:,0]}')
            print(f'epoch:{epoch+1}  loss:{los}') # 
            print(f'before_loss:{before_loss}') ## 一個前のepoch loss 
            torch.autograd.set_detect_anomaly(True)
            los.backward(retain_graph=True)
            running_loss += los.item()
            local_loss.append(los.item())
            del los
            optimizer.step()
            

            # # print statistics
                

        
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(tqdm(test_iter, desc='test')):
                # print(i)
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                los = loss.vector_loss(output, labels)
                test_loss.append(los.item())
                del los
                
                
        

    
        
        
        mean_loss = np.mean(local_loss) 
        before_loss = mean_loss
        print("mean loss",mean_loss)
        loss_hist.append(mean_loss)
        test_mean_loss = np.mean(test_loss) 
        test_hist.append(test_mean_loss)
except:
    traceback.print_exc()
    pass
end_time = time.time()


# ログファイル二セーブ
path_w = 'loss_hist.txt'
with open(path_w, mode='w') as f:
    now = datetime.datetime.now()
    f.write(f'{now}\n')
    for i , x in enumerate(loss_hist):
        f.write(f"{i}: {x}\n")


##　最後の出力結果の確認用
print(output)

## save model
enddir = "models/models_state_dict_end.pth"
torch.save(model.state_dict(), enddir)
print("success model saving")


## analysis

test_iter = DataLoader(test_dataset, batch_size=1, shuffle=True)
analyze(model=model, device=device, test_iter=test_iter, loss_hist=loss_hist,
        test_hist=test_hist, start_time=start_time, end_time=end_time, epoch=epoch,
        lr=lr, tau=args.tau)





