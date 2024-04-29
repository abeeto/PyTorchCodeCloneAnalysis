# Recall that gradient descent can take a long time when processing the entire training set. So it's better to split the training set into batches to process separately.
# Some definitions:
# epoch: one forward and backward pass of ALL training samples
# batch_size: number of training samples used in one forward/backward pass
# number of iterations: number of passes, each pass (forward and backward) using [batch_size] number of samples
# e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch

import enum
from re import M
from matplotlib import transforms
from sklearn.utils import shuffle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# using the wine dataset, each wine can be of 3 classes
class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
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

# custom transform class
# to tensor transform
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# multiplication transform
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, labels = first_data
# print(features, labels)
# print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)

first_data = dataset[0]
features, labels = first_data
# print(features, labels)
# print(type(features), type(labels))
