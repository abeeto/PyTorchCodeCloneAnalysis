import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd.variable import Variable as V
import pandas as pd
import numpy as np

def minmaxscale(data):
    data=(data-data.min())/(data.max()-data.min())
    return data

class LSTM_Model(nn.Module):
    def __init__(self,input_size):
        super(LSTM_Model,self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,#输入维度
            hidden_size=64,#隐藏层数量
            num_layers=4,#网络层数
            dropout=0.2
        )
        self.flat = nn.Flatten()
        self.linear0 = nn.Linear(6400,512)
        self.linear1 = nn.Linear(512,40)

    def forward(self,x):
        result,(h,c) = self.rnn(x)
        result = self.flat(result)
        result = self.linear0(result)
        result = self.linear1(result)
        return result

class TrainData(Dataset):
    def __init__(self):
        self.data0 = pd.read_csv("usdjpy_d.csv",sep=',',header=0)

        self.data0['Date'] = (pd.to_datetime(self.data0['Date']).values-np.datetime64('1991-01-01T00:00:00Z'))/ np.timedelta64(1, 'h')/24

        # self.data0['Open']= minmaxscale(self.data0['Open'])
        # self.data0['High'] = minmaxscale(self.data0['High'])
        # self.data0['Low'] = minmaxscale(self.data0['Low'])
        # self.data0['Close'] = minmaxscale(self.data0['Close'])

        self.data0 = np.array(self.data0)
        self.data0 = torch.tensor(self.data0)
        self.data = self.data0[:-10,1:]
        self.label =self.data0[100: ,1:]
        # print(self.data)
        # print(self.label)

    def __getitem__(self, index):
        return self.data[index:index+100].float(),torch.flatten(self.label[index:index+10].float())

    def __len__(self):
        return len(self.label)-10


trainData = TrainData()
trainDataLoader = DataLoader(trainData,batch_size=64,shuffle=False,pin_memory=True)
lstm = LSTM_Model(4)
lstm = lstm.cuda()

LR = 0.00003
epochs = 4800

optimizer = torch.optim.Adam(lstm.parameters(),lr=LR)

loss_func = nn.MSELoss()
loss_func = loss_func.cuda()

for epoch in range(epochs):
    loss_total = 0
    for input,target in trainDataLoader:
        input=input.cuda()
        target = target.cuda()
        # print('input',input.shape)
        output = lstm(input)
        output = torch.squeeze(output)
        # print('output',output.shape)
        # print('target',target.shape)
        loss = loss_func(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss
    print(epoch,':',(loss_total/len(trainDataLoader)).item())
    if epoch % 100 == 0:
        torch.save(lstm,'lstm0.pkl')
torch.save(lstm,'lstm0.pkl')




