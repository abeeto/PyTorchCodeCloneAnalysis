""" 
Description: We will implement methods to create a dataset using pytorch's Dataset and Dataloader class.
"""
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, [0]])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

winedataset = WineDataset("./wine.csv")
dataloader = DataLoader(winedataset, batch_size=4, shuffle=True, num_workers=4)

# x, y = next(iter(dataloader))
# print(x, y)

# Implementing dummy training loop
EPOCHS = 2
total_samples = len(winedataset)
number_of_steps = math.ceil(total_samples/4)
# print(total_samples, number_of_steps)

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(dataloader):
        if (i+1)%5 == 0:
            print(f'Epoch {epoch+1} :: Training Step  {i+1} :: {x.shape}')
