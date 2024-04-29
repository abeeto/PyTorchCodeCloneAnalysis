import numpy
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=numpy.float32, skiprows=1)

        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1

        self.samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=None)
firstData = dataset[0]
features, labels = firstData
print(features)
print(type(features), type(labels))

# Composing multiple transforms into one
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(10)])
dataset = WineDataset(transform=composed)
firstData = dataset[0]
features, labels = firstData
print(features)
print(type(features), type(labels))
