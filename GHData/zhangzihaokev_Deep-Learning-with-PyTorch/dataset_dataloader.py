# Recall that gradient descent can take a long time when processing the entire training set. So it's better to split the training set into batches to process separately.
# Some definitions:
# epoch: one forward and backward pass of ALL training samples
# batch_size: number of training samples used in one forward/backward pass
# number of iterations: number of passes, each pass (forward and backward) using [batch_size] number of samples
# e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch

import enum
from sklearn.utils import shuffle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# using the wine dataset, each wine can be of 3 classes
class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
data_iter = iter(data_loader)
data = data_iter.next()
features, labels = data
# print(features, labels)

# training loop
n_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(f'total samples = {total_samples}, number of iterations = {n_iterations}')

for epoch in range (n_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        # forward pass
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape}')

# some built in datasets:
torchvision.datasets.MNIST()
