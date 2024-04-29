#!/bin/env python

import torch;
import torch.nn as nn;
import torchvision as tv;
from torch.utils.data import DataLoader;
import matplotlib.pyplot as pyplot;

device = 'cuda';
b_size = 32;

celeb = DataLoader(
	tv.datasets.CelebA(
		root = './dataset/',
		split = 'all',
		download = False,
		target_type = 'attr',
		transform = tv.transforms.Compose([
			tv.transforms.PILToTensor(),
			tv.transforms.Resize(128),
			tv.transforms.CenterCrop(128),
			tv.transforms.Lambda(lambda x:x/256.),
			# tv.transforms.Scale(
			# tv.transforms.CenterCrop(128),
		]),
	),
	batch_size = 32,
	shuffle = True,
	# pin_memory = True,
	# num_workers = 8
);
n_data = len(celeb.dataset);

class BAD(nn.Module):
	def __init__(_, in_channels):
		super().__init__();
		_.bat = nn.BatchNorm2d(in_channels);
		_.act = nn.LeakyReLU();
		_.drp = nn.Dropout();

	def forward(_, x):
		x = _.bat(x);
		x = _.act(x);
		x = _.drp(x);
		return x;

class Gaussian(nn.Module):
	def __init__(_, in_feature, out_feature):
		super().__init__();
		_.out_feat = out_feature;
		_.mean = nn.Linear(in_feature, out_feature);
		_.stdv = nn.Linear(in_feature, out_feature);

	def forward(_, x):
		mu = _.mean(x);
		sd = _.stdv(x);
		eps = torch.randn(_.out_feat).to(device);
		return mu+eps*torch.exp(sd/2.);

class Encoder(nn.Module):
	def __init__(_, ls_dim):
		super().__init__();
		lyr = [];
		conv_in = [3, 32, 64, 64];
		conv_out = [32, 64, 64, 64];
		_.ls_dim = ls_dim;

		for i in range(4):
			lyr.append(nn.Conv2d(
				in_channels = conv_in[i],
				out_channels = conv_out[i],
				stride = 2,
				kernel_size = 3,
				padding = 1
			));
			lyr.append(BAD(
				in_channels = conv_out[i]
			));
		lyr.append(nn.Flatten());
		lyr.append(Gaussian(4096, _.ls_dim));
		_.layer = nn.ModuleList(lyr);

	def forward(_, x):
		# format = "%-35s%s";
		# print(format%(x.size(), "Input"));
		for L in _.layer:
			x = L(x);
			# print(format%(x.size(), L));
		return x;

class Reshape(nn.Module):
	def __init__(_, size):
		super().__init__();
		_.size = size;

	def forward(_, x):
		x = torch.reshape(x, (-1, *_.size));
		return x;

class Decoder(nn.Module):
	def __init__(_, ls_dim):
		super().__init__();
		lyr = [];

		tconv_in = [64, 64, 64, 32]
		tconv_out = [64, 64, 32, 3]

		lyr.append(nn.Linear(ls_dim, 4096));
		lyr.append(Reshape((64, 8, 8)));
		for i in range(3):
			lyr.append(nn.ConvTranspose2d(
				in_channels = tconv_in[i],
				out_channels = tconv_out[i],
				stride = 2,
				kernel_size = 3,
				padding = 1,
				output_padding=1
			));
			lyr.append(BAD(
				in_channels = tconv_out[i]
			));

		lyr.append(nn.ConvTranspose2d(
			in_channels = tconv_in[3],
			out_channels = tconv_out[3],
			stride = 2,
			kernel_size = 3,
			padding = 1,
			output_padding=1
		));
		lyr.append(nn.ReLU());

		_.layers = nn.ModuleList(lyr);

	def forward(_, x):
		# format = "%-35s%s";
		# print(format%(x.size(), "Input"));
		for L in _.layers:
			x = L(x);
			# print(format%(x.size(), L));
		return x;

class VarAE(nn.Module):
	def __init__(_, ls_dim):
		super().__init__();
		_.enc = Encoder(ls_dim);
		_.dec = Decoder(ls_dim);

	def forward(_, x):
		x = _.enc(x);
		x = _.dec(x);
		return x;

	def encode(_, x):
		ls = _.enc(x.unsqueeze(0));
		return ls;

	def decode(_, x):
		img = _.dec(x);
		return img;

class RMSELoss(nn.Module):
	def __init__(_, eps=1e-8):
		super().__init__();
		_.eps = eps;

	def forward(_, y_, y):
		loss = nn.MSELoss();
		return torch.sqrt(loss(y_, y)+_.eps);

class VAELoss(nn.Module):
	def __init__(_):
		super().__init__();
		_.rl = RMSELoss();
		_.kl = nn.KLDivLoss(log_target=True);
		# _.ls = nn.LogSoftmax(im=0);
		# _.ce = nn.CrossEntropyLoss();

	def forward(_, y_, y):
		# return _.rl(y_, y)+_.ls(_.kl(y_, y));
		return _.rl(y_, y)+_.kl(y_, y);

# img = next(iter(celeb))[0].to(device);
# img = torch.randn(32, 3, 128, 128, device=device);
# img.cuda();
# vae = VarAE(ls_dim=100).to(device);
# print(vae(img).size());
# exit();

ls_dim = 200;
epochs = 5;
vae = VarAE(ls_dim=ls_dim).to(device);
optimizer = torch.optim.Adam(params=vae.parameters(), lr=5e-4);
loss_fn = VAELoss();


DEBUG = False;
# Training
for epoch in range(epochs):
	if DEBUG:
		break;
	counter = 0;
	for imgc,_ in celeb:
		img = imgc.to(device);
		restore = vae(img);
		loss = loss_fn(restore, img);

		vae.zero_grad();
		loss.backward();
		optimizer.step();

		counter = counter+b_size;
		print("Epoch: %02d  Loss: %lf Dataset seen: %06d"
			% (epoch, loss, counter),
			end='\r', flush=True);

if DEBUG:
	vae.load_state_dict(torch.load("vae-faces:ep=%d,ls=%d.torch"%(epochs,ls_dim)));
else:
	torch.save(vae.state_dict(), "vae-faces:ep=%d,ls=%d.torch"%(epochs,ls_dim));
	vae.load_state_dict(torch.load("vae-faces:ep=%d,ls=%d.torch"%(epochs,ls_dim)));

# Prepare for evaluation
vae.eval();
vae.to(device);


import PIL;

with PIL.Image.open('./musk.jpg') as i:
	img =  tv.transforms.Compose([
		tv.transforms.PILToTensor(),
		tv.transforms.Resize(128),
		tv.transforms.CenterCrop(128),
		tv.transforms.Lambda(lambda x:1-x/256.),
	])(i);

with torch.no_grad():
	img_res = vae.decode(vae.encode(img.to(device)));

# exit();
from matplotlib import pyplot;
img_res = img_res.cpu();
pyplot.subplot(3,2,1);
pyplot.imshow(img[1], cmap='Reds');
pyplot.subplot(3,2,2);
pyplot.imshow(img_res[0][0], cmap='Reds');
pyplot.subplot(3,2,3);
pyplot.imshow(img[1], cmap='Greens');
pyplot.subplot(3,2,4);
pyplot.imshow(img_res[0][1], cmap='Greens');
pyplot.subplot(3,2,5);
pyplot.imshow(img[1], cmap='Blues');
pyplot.subplot(3,2,6);
pyplot.imshow(img_res[0][2], cmap='Blues');
pyplot.tight_layout();
pyplot.savefig("img:ls=100,ep=10.jpg", dpi=150);

print("Image drawn!");
