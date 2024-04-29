from tarfile import DIRTYPE
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import os 
from tqdm import tqdm
# from rectangle_builder import rectangle,test_img
import sys
# sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
# from model import snu_layer
# from model import network
from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
# import scipy.io
# from torchsummary import summary
import argparse
import h5py

class LoadDataset(Dataset):
    def __init__(self, dir, which:str = "train", time = 100, width = 128, height = 128):
        self.dir = dir
        self.which = which
        #h5ファイルのディレクトリのリスト
        self.dir_h5 = []
        self.width = width
        self.height = height
        self.time = time
        for _, _, files in os.walk(self.dir):
            for file in files:
                if file.endswith('.h5'):
                    self.dir_h5.append(os.path.join(self.dir, file)) 

        "テストデータ"
        self.divide = int((len(self.dir_h5)*0.8))
        if which == "train":
            self.dir_h5 = self.dir_h5[:self.divide]
        elif which == "test":
            self.dir_h5 = self.dir_h5[self.divide:]
        else:
            print("error by data.py")
            exit()
        self.index = len(self.dir_h5)
        
    def __len__(self):
        return self.index

    def __getitem__(self, index):
        events = torch.zeros(2, self.time, self.width, self.height)
        with h5py.File(self.dir_h5[index], "r") as f:
            label = f['label'][()]

            self.events_ = f['events'][()]
            for i_, i in enumerate(self.events_):
                ###i(h5ファイルから読み込まれるデータ):(timestep, y?, x?, pol)
                if i[0] >= self.time:
                    break
                # print(i)
                events[ i[3], i[0], i[2], i[1]] = 1
                ###events:(pol, time, x, y)
                # print("asdfasdadassdfa")
        return events, label

if __name__ == "__main__":
    a = LoadDataset('C:/Users/oosim/Desktop/snn/v2e/output/', time = 20)
    ru = a.__getitem__(2)
    kazu = torch.count_nonzero(ru[0] == 1.)
    print(f'kazu:{kazu}')
    print('1111111111')
    print(a.dir_h5[0])
    print(ru)
    print(ru[0].shape)
    
