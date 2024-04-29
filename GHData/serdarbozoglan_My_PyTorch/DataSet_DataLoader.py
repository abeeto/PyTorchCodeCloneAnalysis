import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd 

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('wine_data.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) # first column 
        self.y = torch.from_numpy(xy[:, [0]]) # first column target
        self.n_samples = xy.shape[0]

        #Data'yi pandas ile okumak istersek:
        #data = pd.read_csv('wine_data.csv', skiprows=1)
        #self.xy = torch.from_numpy(data.to_numpy())
        #self.x = torch.from_numpy(data.iloc[:, 1:].to_numpy())
        #self.y = torch.from_numpy(data.iloc[:, 0].to_numpy())
        #self.n_samples = len(data)

    def __getitem__(self, index):   # datatset[0], calling  a sample from dataset
        # __getitem__ i kullanmak icin X[3] gibi gondermeke yeterli
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]
# first_features, first_label = first_data
# print(first_features, " && ", first_label)
dataloader  = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# the below is for just a one iter.next()
# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data 

# print(features, labels)

# let's create a for loop for whole training set
EPOCH_SIZE = 20
total_samples = dataset.__len__() # veya len(dataset)
n_iterations = math.ceil(total_samples / 4 ) # 4 is batch_size
print(total_samples, n_iterations)

for epoch in range(EPOCH_SIZE):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 ==0:
            print(f'epoch {epoch+1}/{EPOCH_SIZE} step {i+1}/{n_iterations}' ) 



