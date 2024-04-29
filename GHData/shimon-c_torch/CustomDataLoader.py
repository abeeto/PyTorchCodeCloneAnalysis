from __future__ import print_function, division

import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ResNet import norm_data

"""

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                          transform=transforms.Compose(
                                                              [transforms.ToTensor(),
                                                               norm_data]
                                                          )),
                                           batch_size=32, shuffle=True)
"""

class CustomDataLoader(Dataset):
    def __init__(self, file=None, batch_size=32, norm_flag=True):
        if file is None:
            self.data = datasets.MNIST('../data', train=True, download=True,
                                                              transform=transforms.Compose(
                                                                  [transforms.ToTensor(),
                                                                   #transforms.Normalize((0.1307,),(0.308,)),
                                                                   norm_data
                                                                   ]
                                                              ))


        self.x_data = self.data.train_data
        self.y_data = self.data.train_labels
        self.len = self.x_data.shape[0]
        self.batch_size = batch_size
        self.norm_flag = norm_flag
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        L = self.len
        item = item % L
        x = self.x_data[item,...]
        y = self.y_data[item,]
        x = x.unsqueeze(0)
        if self.norm_flag:
            x = norm_data(x)
        return x,y

    def get_loader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size,
                                           shuffle=True)


def mul(x1: float, x2:float) -> float:
    y = x1*x2
    return y

print(f'5*6={mul(5,6)}')
if __name__ == '__main__':
    ld = CustomDataLoader(None)
    ldd = ld.get_loader()
    for bid, (x,y) in enumerate(ldd):
        print(f'bid={bid}, x.shape={x.shape}, y.shape={y.shape}')
        img = x[0,]
        img = img.numpy()
    print('The end')