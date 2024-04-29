import  torch
import  torchvision
from  torch.utils.data import Dataset, DataLoader
import  numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_sample = xy.shape[0]


    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_sample

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features, labels)


