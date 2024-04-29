import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=',',
                dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #return dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        #return len(dataset)
        return self.n_samples

data = WineDataset()
dataloader = DataLoader(dataset=data, batch_size=4,
        shuffle=True, num_workers=2)

# training loop
epochs = 2
total_samples = len(data)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward -> backward -> update
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{epochs}, step {i+1}/{n_iterations},')
