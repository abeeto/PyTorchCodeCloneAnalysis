#!/bin/env python

import torch as t;
import torch.nn as nn;
import torchvision as tv;
import torch.functional as F;
from torch.utils.data import DataLoader;
import numpy as np

NUM_CLASSES = 10;
BATCH_SIZE = 32;

# Load the data
cif = DataLoader(
	tv.datasets.CIFAR10(
		'./dataset/',
		train=True,
		transform = tv.transforms.Compose([
			tv.transforms.PILToTensor(),
			tv.transforms.Lambda(lambda x:x/255.) ]) ),
	batch_size = BATCH_SIZE,
	shuffle = True
	);

cif_test = DataLoader(
	tv.datasets.CIFAR10(
		'./dataset/',
		train=True,
		transform = tv.transforms.Compose([
			tv.transforms.PILToTensor(),
			tv.transforms.Lambda(lambda x:x/255.) ]) ),
	batch_size = BATCH_SIZE,
	shuffle = True
	);

# Build the network
class Model(nn.Module):
	def __init__(self):
		super().__init__();
		self.lg= nn.Sequential (
			nn.Conv2d(3, 32,
				kernel_size = 3,
				padding = 'same'),
			nn.BatchNorm2d(32, momentum=.9),
			nn.LeakyReLU(),

			nn.Conv2d(32, 32,
				kernel_size = 3,
				stride=2),
			nn.BatchNorm2d(32, momentum=.9),
			nn.LeakyReLU(),

			nn.Conv2d(32, 64,
				kernel_size = 3,
				padding = 'same'),
			nn.BatchNorm2d(64, momentum=.9),
			nn.LeakyReLU(),

			nn.Conv2d(64, 64,
				kernel_size = 3,
				stride=2),
			nn.BatchNorm2d(64, momentum=.9),
			nn.LeakyReLU(),

			nn.Flatten(),

			nn.Linear(3136, 128),
			nn.BatchNorm1d(128, momentum=.9),
			nn.LeakyReLU(),
			nn.Dropout(p=.25),

			nn.Linear(128, NUM_CLASSES),
			nn.Softmax(1)
		);

	def forward(self, x):
		# print("%-35s%s"%(x.size(), "Input"));
		for L in self.lg:
			x = L(x);
			# print("%-35s%s"%(x.size(), L));
		return x
# Define a model instance, the optimizer and loss
model = Model();
optimizer = t.optim.Adam(model.parameters(), lr=5e-4);
loss_fn = nn.CrossEntropyLoss();

# Train
for epoch in range(10):
	counter = 0;
	for xb, yb in cif:
		y = model(xb);
		loss = loss_fn(y, yb);

		loss.backward();
		optimizer.step();
		optimizer.zero_grad();
		counter = t.min(t.tensor(
			[len(cif.dataset), counter+BATCH_SIZE]));
		print("Epoch: %02d  Loss: %lf Epoch completion: %.2lf%%"%
			(epoch,loss, counter/500),
			end='\r', flush=True);

	print('',end='\n');

# Test
acc = 0;
count = 0;
for xb, yb in cif_test:
	y = t.argmax(model(xb), dim=1);
	acc += t.count_nonzero(t.eq(y, yb)).item();
	count += BATCH_SIZE;
	print("Testing completion: %.2lf%%"
		%(count*100./len(cif_test.dataset)),
		flush=True, end='\r'
	);

print("\nAccuracy on test set: %.3lf"%(acc/len(cif_test.dataset)));



