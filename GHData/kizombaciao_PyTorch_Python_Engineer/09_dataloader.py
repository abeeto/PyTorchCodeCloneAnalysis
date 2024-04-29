import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# batch
# 100 samples, batch_size = 20 => 100 / 20 = 5 iterations for 1 epoch

# DataLoader can do the batch computation for us
# Implement a custom Dataset:
# inherit Dataset
# implement __init__, __getitem__, and __len__

class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

    # basically, this allows your objects to be indexed !!!
    # here we are allowing the object to index tuple data !
    def __getitem__(self, index):                           
        return self.x_data[index], self.y_data[index]

    def __len__(self):                                      
        return self.n_samples

# create dataset
dataset = WineDataset()

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data   # how does it know which special method ???
print(features, labels)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0, instead of 2 !!!
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(f'{features} \n {labels}')

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)

dataiter = iter(train_loader)
data = next(dataiter)
x, y = data
print(x.shape, y.shape)


