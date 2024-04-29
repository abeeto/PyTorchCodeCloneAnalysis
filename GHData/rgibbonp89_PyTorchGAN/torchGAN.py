# a simple MLP GAN (no conv layers or the like)

import numpy as np
import os
import cv2

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms
import torch.autograd as autograd
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

import torch.nn.functional as F
import torch.optim as optim

import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'GAN implementation in PyTorch')

parser.add_argument('--number', type = float, default = 3, help = 'GAN number')
parser.add_argument('--epochs', type = float, default = 4, help = 'Number of epochs')
parser.add_argument('--glr', type = float, default = .00001, help = 'Generator lr')
parser.add_argument('--dlr', type = float, default = .001, help = 'Discriminator lr')
parser.add_argument('--mnist', type = str, default = 'digits', choices = ['digits', 'fashion'], help = 'Dataset')


args = parser.parse_args()
 
def mnist_loader(numbers):
	def one_hot(label, output_dim):
		one_hot = np.zeros((len(label), output_dim))	
		for idx in range(0,len(label)):
			one_hot[idx, label[idx]] = 1		
		return one_hot
	#Training Data
	f = open('./data/train-images.idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) /  127.5 - 1
	f = open('./data/train-labels.idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	newtrainX = []
	for idx in range(0,len(trainX)):
		if trainY[idx] in numbers:
			newtrainX.append(trainX[idx])
	return np.array(newtrainX), trainY, len(trainX)

def fashion_mnist_loader(numbers):
	def one_hot(label, output_dim):
		one_hot = np.zeros((len(label), output_dim))	
		for idx in range(0,len(label)):
			one_hot[idx, label[idx]] = 1		
		return one_hot
	#Training Data
	f = open('/home/richard/Desktop/QuantFriday/gan_numpy/data/fashion/train-images-idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) /  127.5 - 1
	f = open('/home/richard/Desktop/QuantFriday/gan_numpy/data/fashion/train-labels-idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	newtrainX = []
	for idx in range(0,len(trainX)):
		if trainY[idx] in numbers:
			newtrainX.append(trainX[idx])
	return np.array(newtrainX), trainY, len(trainX) 


class Generator(nn.Module):
	def __init__(self, dataset = 'mnist'):
		super(Generator, self).__init__()
		if dataset == 'mnist' or dataset == 'fashion-mnist':
			self.input_height = 28
			self.input_width = 28
			self.input_dim = 100
			self.output_dim = 1
			# make the first layer fc mapping 100 to 1028
			# 103828 parameters
			# leaky relu activation
		self.fc1 = nn.Sequential(
			nn.Linear(self.input_dim, 1028),
			nn.LeakyReLU(.02)
			)
			# similar deal here - fc from 1028 to 128
			# 103828 + 131712 params so far
		self.fc2 = nn.Sequential(
			nn.Linear(1028, 128),
			nn.LeakyReLU(.02)
			)
			# finally we have our output layer
			# map fc to 28*28 (but flattened) = same dimensions as image
			# 103828 + 131712 + 101136 params
		self.fc3 = nn.Sequential(
			nn.Linear(128, 784),
			nn.LeakyReLU(.02)
			)	
	def forward(self, input):
		x = self.fc1(input)
		x = self.fc2(x)
		x = self.fc3(x)
		x = torch.sigmoid(x) # map to [0,1] using sigmoid, corresponding to a black and white pixel intensity
		x = x.view(-1, 28 ,28) # reshape 784*1 vector to 28*28
		return x



class Discriminator(nn.Module):
	def __init__(self, dataset = 'mnist'):
		super(Discriminator, self).__init__()
		if dataset == 'mnist' or dataset == 'fashion-mnist':
			self.input_height = 28
			self.input_width = 28
			self.input_dim = 784
			self.output_dim = 1
			# take flattened input image and pass it through a very simple fc network
		self.fc1 = nn.Sequential(
			nn.Linear(self.input_dim, 128),
			nn.LeakyReLU(.02))
			# output of a size 1 - corresponds to raw logit for whether image is real
		self.fc3 = nn.Sequential(
			nn.Linear(128, 1)
			)
	def forward(self, input):
		x = input.view(-1, 784)
		x = self.fc1(x)
		x = self.fc3(x)
		x = torch.sigmoid(x) # pass through sigmoid - interpreted as probability that given object is real
		return x

# load data
if args.mnist == 'digits':
	trainx, _, train_size = mnist_loader([args.number])
else:
	trainx, _, train_size = fashion_mnist_loader([args.number])

# instantiate objects of the class Generator and Discriminator respectively
G = Generator()
D = Discriminator()


g_loss = nn.BCELoss()
d_loss = nn.BCELoss()

# best results when lrG = .00001 and lrD = .001

lrG = args.glr
lrD = args.dlr



for j in range(int(args.epochs)):
	lrG = lrG * (1/(1 + .001*j)) # lr decay
	lrD = lrD * (1/(1 + .001*j))
	G_optim = optim.Adam(G.parameters(), lrG, (.99,.999)) # define optimizers separately
	D_optim = optim.Adam(D.parameters(), lrD, (.99,.999))
	for i in range(int(6742//4 - 1)):
				# data
		datapass = Variable(torch.from_numpy(np.array(trainx[i:i+16]))) # grab batch of 16 real images
		z = torch.FloatTensor(16, 100).uniform_(-1,1) # generate 16 100-length arrays from U(-1,1)
		# forward pass
		out = G(Variable(z)) # use generator to create fake images from U(-1,1) data
		fake_d = D(out) # give probabilities that each fake image is real
		true_d = D(datapass) # same for real images
		# calculate D loss
		y_real, y_fake = Variable(torch.ones(16, 1)), Variable(torch.zeros(16, 1)) # 1,0 labels for fake and real images
		D_optim.zero_grad() # reset optimizer
		dl_real = d_loss(true_d, y_real) # calculate loss for real images
		dl_fake = d_loss(fake_d, y_fake)
		dl = dl_real + dl_fake
		dl.backward(retain_graph = True)
		# calculate G loss
		G_optim.zero_grad()
		gl = g_loss(fake_d, y_real)
		gl.backward(retain_graph = False)
		G_optim.step()
		D_optim.step()
		print([gl, dl, i])


z = torch.FloatTensor(1, 100).uniform_(-1,1)
out = G(Variable(z))

import matplotlib.pyplot as plt

im = plt.imshow(out[0].data.numpy(), cmap='gray', interpolation='none')
cbar = plt.colorbar(im)
plt.show()

# real number

im_real = plt.imshow(trainx[1].reshape(28,28), cmap='gray', interpolation='none')
cbar = plt.colorbar(im_real)
plt.show()