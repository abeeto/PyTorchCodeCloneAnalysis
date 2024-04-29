from random import shuffle
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
import pandas as pd
import argparse
import h5py
import time
from PIL import Image
import cv2
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
        # print(self.dir_h5)
        
    def __len__(self):
        return self.index

    def __getitem__(self, index):
        events = torch.zeros(2, self.time, self.width, self.height)
        events_p = torch.zeros(self.time, self.width, self.height)
        events_n = torch.zeros(self.time, self.width, self.height)
        with h5py.File(self.dir_h5[index], "r") as f:
            label = f['label'][()]

            self.events_ = f['events'][()]
            for i_, i in enumerate(self.events_):
                ###i(h5ファイルから読み込まれるデータ):(timestep, y?, x?, pol)
                if i[0] >= self.time:
                    break
                # print(i)
                events[ i[3], i[0], i[2], i[1]] = 1
                if i[3]:
                    # print(1)
                    events_p[i[0], i[2], i[1]] = 1
                else:
                    # print(0)
                    events_n[i[0], i[2], i[1]] = 1
                ###events:(pol, time, x, y)
                # print("asdfasdadassdfa")
        return events_p, events_n

if __name__ == "__main__":
    a = LoadDataset('C:/Users/oosim/Desktop/snn/v2e/output3/', time = 20,)
    a = LoadDataset('C:/Users/oosim/Desktop/snn/v2e/output_vector/', time = 20,)
    # (time, x, y)
    num = int(input())
    # print(a)
    p,n  = a[num]
    # e = torch.zeros_like(p)
    # p_ = torchvision.transforms.functional.to_pil_image(p[11,:,:]*255)
    # p_.show()
    images = []
    time_ = 20
    x, y = 128, 128
    channel = 3
    events = torch.zeros(time_, channel, x, y)
    for i in range(20):
        events[i,0,:,:] =  p[i,:,:]*300
        events[i,1,:,:] =  n[i,:,:]*300
        #Red:positive
        #Green:negative
        p_ = torchvision.transforms.functional.to_pil_image(events[i,:,:, :])
        
        # p_true = p[i,:,:]==1
        # n_true = n[i,:,:]==1
        # print(f'count:{p_true.sum() + n_true.sum()}')

        images.append(p_)
    images[0].save('youtube/douga.gif', duration = 500, save_all=True, append_images=images[1:], loop = 50)
    

    gif = cv2.VideoCapture('youtube/douga.gif')
    fps = gif.get(cv2.CAP_PROP_FPS)  # fpsは１秒あたりのコマ数
    
    images = []
    i = 0
    while True:
        is_success, img = gif.read()
        if not is_success:
            break

        images.append(img)
        i += 1
    cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)

    for t in range(len(images)):
        cv2.imshow('test', images[t])
        cv2.waitKey(int(1000/fps))  # １コマを表示するミリ秒

    cv2.destroyAllWindows()