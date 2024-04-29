# import dependencies
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# INEFFICIENT METHOD
# loading the entire data set for training like this can be computationally expensive
#*************************************************************************************

#data = np.loadtxt("./data/wine.csv")
# training loop
# for epoch in range(1000):
#     x, y = data


# EFFICIENT METHOD
# loading data in batches is more computationally efficient
#*************************************************************************************
# training loop
# for epoch in range(1000):
#     # loop over all batches
#     for i in range(total_batches):
#         x_batch, y_batch = "....."


# PYTORCH'S DATASET AND DATALOADER LETS US DO THIS EFFICIENTLY
#*************************************************************************************
# Definition of Terms
#epoch = 1 forward and backward pass of ALL training samples

#batch_size = number of training samples in one forward and backward pass

#number of iterations = number of batches that will use up all the number of samples

#e.g. 100 samples, batch_size=20 ---> 100/20 = 5 iterations of 1 epoch


# Implementation of Our Own Custom Dataset
class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt("./data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # this will allow for indexing and returns a tuple of features and targets
        # so we can call dataset[0] for example
        return self.x[index], self.y[index]

    def __len__(self):
        # this will allow us to call len(dataset)
        return self.n_samples



dataset = WineDataset()
print(len(dataset))
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# we can now convert data loader into an iterator
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# we can iterate over the entire dataloader in training loop
# training loop
num_epochs = 2
total_samples = len(dataset)
batch_size = 4
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples, n_iterations)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 == 0:
            # i loops through the number of batches and ends when on n_iterations
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}")


# datasets
# torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco






