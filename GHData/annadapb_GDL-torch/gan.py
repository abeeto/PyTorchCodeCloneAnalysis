#!/bin/env python

import numpy as np;
from torchvision import transforms as tf;

# Import the dataset
class QuickDraw():
	def __init__(_, transforms):
		_.data = np.load('./dataset/full_numpy_bitmap_camel.npy');
		_.length = len(_.data);
		_.data.resize((_.length, 28, 28));
		_.data = transforms(_.data);

	def __len__(_):
		return _.length;

	def __getitem__(_, i):
		return _.data[i];

	def __str__(_):
		return '''Quick, Draw! dataset by Google.
Length: %d\nSize: %s'''%(_.length, _.data.shape);

from torch.utils.data import DataLoader;
BATCH_SIZE = 32;
quickdraw = DataLoader(
		dataset = QuickDraw(transforms = tf.Compose([
			tf.Lambda(lambda x:x/256.),
		])),
		shuffle = True,
		batch_size = BATCH_SIZE,
	);

print(quickdraw.dataset);

# Build the network
import torch;
from torch import nn;
import torchvision as tv;

class ADConv2d(nn.Module):
	def __init__(_,channels,kernel,stride,
	momentum = .1, dropout = .5):
		super().__init__();
		num = len(channels);
		if(len(kernel)!=num-1 or len(stride)!=num-1):
			raise ValueError("Kernel or stride length do not "+
				"match with the channels size.\n"+
				"\tchannel(%d-1): kernel(%d) and stride(%d).\n"
				%(num, len(kernel), len(stride) ));
		_.layers = nn.ModuleList();
		for i in range(num-1):
			in_chan = channels[i];
			out_chan = channels[i+1];
			kern = kernel[i];
			strd = stride[i];
			_.layers.append(nn.Conv2d(
				in_channels = in_chan,
				out_channels = out_chan,
				kernel_size = kern,
				stride = strd,
				padding = 1,
			));
			_.layers.append(nn.ReLU());
			_.layers.append(nn.Dropout(p=dropout));

	def forward(_,x):
		for L in _.layers:
			x = L(x);
		return x;

	def debug(_):
		fmt = "%-35s%s";
		x = torch.randn(32, 1, 28, 28);
		print(fmt%(x.size(), "Input()"));
		for L in _.layers:
			x = L(x);
			print(fmt%(x.size(),L));

class Reshape(nn.Module):
	def __init__(_, size):
		super().__init__();
		_.size = size;

	def forward(_, x):
		x = torch.reshape(x, (-1, *_.size));
		return x;

class Discriminator(nn.Module):
	def __init__(_):
		super().__init__();
		_.layers = nn.ModuleList();
		lyr = ADConv2d(
			channels=[1, 64, 64, 128, 128],
			kernel = [3, 3, 3, 3],
			stride = [2, 2, 2, 1]);
		_.layers.append(lyr);
		_.layers.append(nn.Flatten());
		_.layers.append(nn.Linear(2048, 1));

	def forward(_, x):
		for L in _.layers:
			x = L(x);
		return x;

	def debug(_, size=(32, 1, 28, 28)):
		fmt = "%-35s%s";
		x = torch.randn(size);
		print(fmt%(x.size(), "Input()"));
		for L in _.layers:
			x = L(x);
			if 'debug' in dir(L):
				L.debug();
			else:
				print((fmt%(x.size(),L)));

di = Discriminator();
# di.debug();

class Generator(nn.Module):
	def __init__(_):
		super().__init__();
		conv_chan = [64, 128, 64];
		_.layers = nn.ModuleList();
		_.layers.append(nn.Linear(100, 3136));
		_.layers.append(nn.BatchNorm1d(3136, momentum=.9));
		_.layers.append(nn.ReLU());
		_.layers.append(Reshape((64, 7, 7)));
		for i in range(len(conv_chan)-1):
			if 2>i:
				_.layers.append(nn.Upsample(scale_factor=2));
			_.layers.append(nn.Conv2d(
				in_channels  = conv_chan[i],
				out_channels = conv_chan[1+i],
				kernel_size  = 3,
				stride       = 1,
				padding      = 1,
			));
			_.layers.append(nn.BatchNorm2d(conv_chan[1+i]));
			if 4!=i:
				_.layers.append(nn.ReLU());
		_.layers.append(nn.Conv2d(
			in_channels  = 64,
			out_channels = 1,
			kernel_size  = 3,
			stride       = 1,
			padding      = 1,
		));
		_.layers.append(nn.Tanh());

	def forward(_, x):
		for L in _.layers:
			x = L(x);

	def debug(_, size=(32, 100)):
		fmt = "%-35s%s";
		x = torch.randn(size);
		print(fmt%(x.size(), "Input()"));
		for L in _.layers:
			x = L(x);
			if 'debug' in dir(L):
				L.debug();
			else:
				print((fmt%(x.size(),L)));

gene = Generator();
gene.debug();
