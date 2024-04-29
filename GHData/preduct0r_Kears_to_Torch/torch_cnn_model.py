import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn import Sequential
from torch.nn import functional as F
from torch import nn
import numpy as np
import pandas as pd
from copy import deepcopy

# from sampler import BalancedBatchSampler


class Config:
    def __init__(self, lr, batch_size, n_classes, num_epochs=500):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_classes = n_classes

class My_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx, np.newaxis, :]), self.y[idx]

class Batcher:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.classes = np.unique(self.y)
        self.indexes_per_class = {}
        for cls in self.classes:
            self.indexes_per_class[cls] = np.where(self.y==cls)[0].tolist()
        self.each_cls_num = self.batch_size//len(self.classes)
        self.iter = 0

    def get(self):
        if self.iter+1 > self.x.shape[0]//self.batch_size:
            self.iter = 0
        mask=[]
        for idx, cls in enumerate(self.classes):
            mask += self.indexes_per_class[cls][self.iter*self.each_cls_num:(self.iter+1)*self.each_cls_num]
            # self.indexes_per_class[cls] = self.indexes_per_class[cls][self.each_cls_num:]
        xb, yb = self.x[mask,:], self.y[mask]
        self.iter+=1
        return (torch.FloatTensor(xb[:, np.newaxis, :]), torch.LongTensor(yb))



class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(round(100 * gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


class torch_model(nn.Module):
    def __init__(self, config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8)):
        super(torch_model, self).__init__()
        self.config = config
        self.fc1 = nn.Conv1d(in_channels= 1, out_channels=8, kernel_size=k_size[0])      #(batch_size, in_channels, seq_len)
        self.bn_1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(kernel_size=p_size[0])

        self.fc2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=k_size[1])
        self.bn_2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(kernel_size=p_size[1])

        self.fc3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=k_size[2])
        self.bn_3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool1d(kernel_size=p_size[2])

        self.fc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k_size[3])
        self.bn_4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool1d(kernel_size=p_size[3])                           #(batch_size, in_channels, seq_len)


        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.gap = nn.AdaptiveAvgPool1d(output_size = 1)
        self.dropout = nn.Dropout(p=0.9)
        self.linear = nn.Linear(128, self.config.n_classes)
        self.bn_final = nn.BatchNorm1d(self.config.n_classes)


    def forward(self, x, hidden):
        out = self.fc1(x)
        out = self.bn_1(out)
        out = self.relu1(out)
        out = self.mp1(out)

        out = self.fc2(out)
        out = self.bn_2(out)
        out = self.relu2(out)
        out = self.mp2(out)

        out = self.fc3(out)
        out = self.bn_3(out)
        out = self.relu3(out)
        out = self.mp3(out)

        out = self.fc4(out)
        out = self.bn_4(out)
        out = self.relu4(out)
        out = self.mp4(out)
        out = out.permute(0,2,1)

        out, hidden = self.lstm1(out, hidden)
        out, hidden = self.lstm2(out, hidden)

        out = out.permute(0, 2, 1)
        out = self.gap(out).squeeze()
        out = self.dropout(out)
        out = self.linear(out)
        out = self.bn_final(out)
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        # weight = next(self.parameters()).data
        hidden = (torch.zeros((1, batch_size, 128)).to('cuda'),
                      torch.zeros((1, batch_size, 128)).zero_().to('cuda'))

        return hidden