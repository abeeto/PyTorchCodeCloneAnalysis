import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# epoch = 1 forward and backward pass of ALL training samples
# batch_size = number of training samples in one forward & backward pass
# number of iterations = number of passes, each pass using [batch_size] number of samples
# e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iteration for 1 epoch
# --> DataLoader can do the batch computation for us

class WineDataset(Dataset):
    def __init__(self):
        # Initialize data, download, etc. , can read with numpy or pandas
        # delimiter = "," because comma seperated our file, skipped 1st row because this is our header
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype = np.float32, skiprows = 1)
        
        self.x = torch.from_numpy(xy[:, 1:]) # converted to tensor from numpy
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples

# create dataset
dataset = WineDataset()

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# convert to an iterator and look at one random sample
datatiter = iter(dataloader)
data = datatiter.next()
features, labels = data
print(features, labels)

# Dummy Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}☺/{n_iterations}, inputs {inputs.shape}')
