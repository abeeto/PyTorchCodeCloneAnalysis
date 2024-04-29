import torch
import os,glob
import random,csv
from    torch.utils.data import Dataset,DataLoader
from    torchvision import transforms
from    PIL import Image
import time
import numpy as np

class Data(Dataset):
    def __init__(self, guard, mode):
        super(Dataset, self).__init__()
        self.root = 'data/'+ guard +'_' +mode+'.txt'
        context=np.loadtxt(self.root,dtype='float32')
        self.labels=context[:,:2]
        img_xyxy=context[:,2:]  # xyxy
        img_w=(img_xyxy[:,2]-img_xyxy[:,0]).reshape(-1,1)
        img_h = (img_xyxy[:, 3] - img_xyxy[:, 1]).reshape(-1,1)
        self.img_center=np.hstack((img_xyxy[:,0].reshape(-1,1)+img_w/2,img_xyxy[:,3].reshape(-1,1),img_w,img_h))
        self.input=self.img_center

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):

        input, label = self.input[idx], self.labels[idx]

        input = torch.tensor(input)
        label = torch.tensor(label)

        return input, label
