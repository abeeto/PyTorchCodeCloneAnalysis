""" 
Description: We will implement methods to create a dataset using pytorch's Dataset and Dataloader class.
"""
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from torchvision import transforms

class WineDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = data[:, 1:]
        self.y = data[:, [0]]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.x.shape[0]


# Lets implement our custom convert_to_tensor transfrom
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


winedataset = WineDataset("./wine.csv")
x, y = winedataset[0]
print(type(x), type(y))
print(x, y)

winedataset = WineDataset("./wine.csv", transform=ToTensor())
x, y = winedataset[0]
print(type(x), type(y))
print(x, y)

winedataset = WineDataset("./wine.csv", transform=transforms.Compose([ToTensor(), MulTransform(2)]))
x, y = winedataset[0]
print(type(x), type(y))
print(x, y)
