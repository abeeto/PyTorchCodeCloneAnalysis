#!/bin/env python

# GDL Chapter 2 Example 1

import torch;
import torchvision as tv;
from torch.utils.data import DataLoader;
import torch.nn as nn;

# 1 Load CIFAR-10 dataset
# - 50,000 image of dim 35x35
# - convert the PIL images into Torch tensors
# - scales values from -1 to 1
cif_train = tv.datasets.CIFAR10(
	'dataset',
	train = True,
	transform=tv.transforms.Compose([
		tv.transforms.PILToTensor(),
		tv.transforms.Lambda(lambda x: x/256.)
	]));
cif_test = tv.datasets.CIFAR10(
	'dataset',
	train = False,
	transform=tv.transforms.Compose([
		tv.transforms.PILToTensor(),
		tv.transforms.Lambda(lambda x: x/256.)
	]));
NUM_CLASSES = 10;
BATCH_SIZE = 32;

cif = DataLoader(cif_train, batch_size=BATCH_SIZE, shuffle=True);
test = DataLoader(cif_test, batch_size=BATCH_SIZE, shuffle=True);

# Data has been loaded as an iterable.
# To access the data, we can use
# 	x = next(iter(cif))  --> tensor( tensor(C, W, H), tensor(lable) )
# 	x[0][0, 12, 13] --> red pixel value at 12 row and 13 col

# xb,yb = next(iter(cif));
# print(xb.size());

'''
model = nn.Sequential(
	nn.Flatten(),
	nn.Linear(3*32*32, 200),
	nn.ReLU(),
	nn.Linear(200, 150),
	nn.ReLU(),
	nn.Linear(150, 10),
	nn.Softmax(1)
);
'''


class model(nn.Module):
	def __init__(self):
		super().__init__();
		self.layer_group = nn.Sequential(
			nn.Flatten(),
			nn.Linear(3072, 200),
			nn.ReLU(),
			nn.Linear(200, 150),
			nn.ReLU(),
			nn.Linear(150, 10),
			# nn.ReLU(),
			nn.Softmax(1),
		);

	def forward(self, x):
		for L in self.layer_group:
			x = L(x);
		return x;

# train, test = next(iter(cif));
# print(model(train).size(), test.size(), _test.size());
# print(model(train), test);
# pred = model()(train);
# print(pred);
# exit();

model1 = model();
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.0005);

for epoch in range(10):
	counter=0;
	best = 100;
	for xb,yb in cif:
		pred = model1(xb);
		loss = loss_fn(pred, yb);

		if (loss.item()<best):
			best = loss.item();
			torch.save(model1.state_dict(), 'best-model');

		loss.backward();
		optimizer.step();
		optimizer.zero_grad();
		counter += xb.size()[0];

		print("Epoch: %02d  Loss: %lf Epoch completion: %02d%%"%
			(epoch,loss, counter//500),
			end='\r', flush=True);

	print("\n", end='');

mod = model();
mod.load_state_dict(torch.load('best-model'));

total_loss = 0;
n = 0;

for xb,yb in test:
	pred = mod(xb);
	loss = loss_fn(pred, yb);
	n += 1;
	total_loss += loss.item();

print("\nTotal evaluation loss: %.6lf"%(total_loss/n));
