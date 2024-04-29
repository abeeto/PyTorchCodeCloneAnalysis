# This code is to construct data loader, and iterate on it

import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader

# Build up data class, which load real data from file.
class DiabetesDataset(Dataset):

	def __init__(self):
		xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',',dtype=np.float32)
		self.len = xy.shape[0]
		self.len = 10
		self.x_data = torch.from_numpy(xy[0:10,0:-1])
		self.y_data = torch.from_numpy(xy[0:10,[-1]])

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

# Create an data instance, which iclude all informationi
dataset = DiabetesDataset()
# Use "DataLoader function to create an iterable train_loader, we can specify the information from dataset, batch size, shuffle"
train_loader = DataLoader(dataset = dataset, batch_size = 3, shuffle = True, num_workers = 1)


# In the training cycle, we loop over epoch, in each epoch, we loop over each mini batch in train_loader.
print("Loop 1\n")
for epoch in range(2): # two epoch
	for i, data in enumerate(train_loader, 0): # loop over all mini batch, print out iterator i and data, iterator starting from 0
		print(i, data)

print("\nLoop 2\n")

# data as defined in class, has two items coupled, one is x, one is y, in order to separate them, we do: " input, labels = data"
for i, data in enumerate(train_loader,0): # print out iterator and data together, iterator from 0
	inputs, labels = data
	print(i, inputs, labels)

print("\nLoop 3\n")
for data in enumerate(train_loader,0): # print out iterator and data together, iterator from 0
	print(data)

print("\nLoop 4\n")
for data in enumerate(train_loader, 20): # print out iterator and data together, iterator from 20
	print(data)