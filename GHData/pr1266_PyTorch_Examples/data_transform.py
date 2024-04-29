import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import os

os.system('cls')

class WineDataset(Dataset):

    def __init__(self, transform = None):
        
        xy = np.loadtxt('wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples

class ToTensorClass:
    def __call__(self, sample):
        inp, out = sample
        return torch.from_numpy(inp), torch.from_numpy(out)

class MulTransformClass:

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inp, out = sample
        inp *= self.factor
        return inp, out

dataset = WineDataset(transform = ToTensorClass())
first_data = dataset[0]
features, label = first_data
print(features, label)
print(type(features), type(label))

composed = torchvision.transforms.Compose([ToTensorClass(), MulTransformClass(2)])
dataset = WineDataset(transform = composed)
first_data = dataset[0]
features, label = first_data
print(features, label)
print(type(features), type(label))
