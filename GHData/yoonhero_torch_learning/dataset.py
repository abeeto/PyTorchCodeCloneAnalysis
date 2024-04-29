from regex import F
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import math

class WindDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt("./data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform
    
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples
    
class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self,factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WindDataset(transform=None)
first_data = dataset[0]
features, labels = first_data

print(type(features), features, labels)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WindDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data

print(type(features), features, labels)

