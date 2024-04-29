#!/bin/env python

import torch as t;
import torchvision as tv;
import torch.nn as nn;
from torch.optim import Adam;
from torch.utils.data import DataLoader;

BATCH_SIZE = 32;
mnist = DataLoader(
		tv.datasets.MNIST(
			root='./dataset/',
			train = True,
			# download = True,
			transform = tv.transforms.Compose([
				    tv.transforms.PILToTensor(),
				    tv.transforms.Lambda(lambda x:x/256.)
			])
		),
		batch_size = BATCH_SIZE,
		shuffle = True
);

n_data = len(mnist.dataset);
# image, _ = next(iter(mnist));
# print("Dataset size{\n\timage: %s,\n\tlabel: %s\n}"
# 	%(image.size(), _.size()) 	);

class Gaussian(nn.Module):
	def __init__(_, input):
		super().__init__();
		_.input = input;

	def forward(_, mu, var):
		eps = t.randn(_.input);
		return mu+eps*t.exp(var/2.);


class Encoder(nn.Module):
	def __init__(_, ls_out):
		super().__init__();
		conv_out     = [32, 64, 64, 64];
		conv_in      = [1, 32, 64, 64];
		conv_kernel  = [3, 3, 3, 3];
		conv_stride  = [1, 2, 2, 1];

		conv_layers = [];
		n_conv_layer = len(conv_out);
		for i in range(n_conv_layer):
			conv_layers.append(nn.Conv2d(
				in_channels  = conv_in[i],
				out_channels = conv_out[i],
				kernel_size  = conv_kernel[i],
				stride       = conv_stride[i],
				padding = 'same' if conv_stride[i]==1 else 1
			));
			conv_layers.append(nn.LeakyReLU())
		conv_layers.append(nn.Flatten());
		# conv_layers.append(nn.Linear(3136, ls_out));

		_.layers = nn.ModuleList(conv_layers);
		_.mean   = nn.Linear(3136, ls_out);
		_.std    = nn.Linear(3136, ls_out);
		_.norm   = Gaussian(input=ls_out);

	def forward(_, x):
		for L in _.layers:
			x = L(x);
			# print(x.size());
		x_mu = _.mean(x);
		x_vr = _.std(x);
		x = _.norm(x_mu, x_vr);
		return x;

# img = t.zeros(BATCH_SIZE, 1, 28, 28);
# dec = Encoder(2)(img);
# print(dec.size());
# exit();

class Reshape(nn.Module):
	def __init__(_, size):
		super().__init__();
		_.size = size;

	def forward(_, x):
		x = t.reshape(x, (-1, *_.size));
		return x;

class Decoder(nn.Module):
	def __init__(_, ls_out):
		super().__init__();
		tconv_in     = [64, 64, 64, 32];
		tconv_out    = [64, 64, 32, 1];
		tconv_kernel = [3, 3, 3, 3];
		tconv_stride = [1, 2, 2, 1];

		layers = [];
		n_layer = len(tconv_out);

		layers.append(nn.Linear(ls_out, 3136));
		layers.append(Reshape([64, 7, 7]));
		for i in range(n_layer):
			layers.append(nn.ConvTranspose2d(
				in_channels    = tconv_in[i],
				out_channels   = tconv_out[i],
				kernel_size    = tconv_kernel[i],
				stride         = tconv_stride[i],
				padding        = 1,
				output_padding = 1 if tconv_stride[i]>1 else 0
			));
			layers.append(nn.LeakyReLU() if i!=n_layer-1 else
					   nn.Sigmoid()   )

		_.layers = nn.ModuleList(layers);

	def forward(_, x):
		for L in _.layers:
			x = L(x);
			# print("%-30s : %s"%(x.size(), L), flush=True);
		return x;

# img = t.zeros(BATCH_SIZE, 2);
# dec = Decoder()(img);
# exit();

class AutoEncoder(nn.Module):
	def __init__(_, lsdim):
		super().__init__();
		_.enc = Encoder(lsdim);
		_.dec = Decoder(lsdim);

	def forward(_, x):
		x = _.enc(x);
		x = _.dec(x);
		return x;

	def encode(_, x):
		with t.no_grad():
			ls = _.enc(x);
		return ls;

	def decode(_, y):
		with t.no_grad():
			img = _.dec(y);
		return img;

	def __str__(_):
		return str(_.enc)+"\n"+str(_.dec);

class RMSELoss(nn.Module):
	def __init__(_, eps=1e-8):
		super().__init__();
		_.eps = eps;

	def forward(_, y_, y):
		loss = nn.MSELoss();
		return t.sqrt(loss(y_, y)+_.eps);

class VARLoss(nn.Module):
	def __init__(_):
		super().__init__();
		_.rl = RMSELoss();
		_.kl = nn.KLDivLoss();

	def forward(_, y_, y):
		return _.rl(y_, y)+_.kl(y_, y);

latent_space_dim = 5;
ae = AutoEncoder(lsdim=latent_space_dim);
img = t.zeros(BATCH_SIZE, 1, 28, 28);
ls  = ae.encode(img);
print("Latent space dimensions: ", ls[-1].size());
ls = t.zeros(1, latent_space_dim);
img = ae.decode(ls);
print("Image dimensions: ", img[-1,-1].size());

# Optimizer
optimizer = Adam(ae.parameters(), lr=5e-4);
loss_fn = VARLoss();

# Training
ae.train();
for e in range(0):
	counter = 0;
	for xb,_ in mnist:
		restore = ae(xb);
		loss = loss_fn(restore, xb);

		loss.backward();
		optimizer.step();
		optimizer.zero_grad();

		counter = t.min(t.tensor(
			[n_data, counter+BATCH_SIZE]));
		print("Epoch: %02d  Loss: %lf Epoch completion: %.2lf%%"
			% (e,loss, counter*100/n_data),
			end='\r', flush=True);

	print('',end='\n');

# t.save(ae.state_dict(), 'vae:ls=5,ep=1.torch');
ae.load_state_dict(t.load('vae:ls=5,ep=1.torch'));

# Latent space
ae.eval();


from matplotlib import pyplot;
col = ['blue', 'orange', 'green', 'red', 'purple',
'brown', 'pink', 'gray', 'olive', 'cyan',]
print('[Generating latent space image] ', end='');
pyplot.figure(figsize=(15, 15));
mnist_data = iter(mnist);
for x in range(10):
	img,label = next(mnist_data);
	ls = ae.encode(img);

	for i in range(len(label)):
		pyplot.scatter(ls[i,0], ls[i,1],
			color=col[label[i].item()], marker='x');

pyplot.savefig('latent-space.png',
		dpi=150,
		bbox_inches='tight',
	);
print('Done.');

# Compare images from input and generator
img,_ = next(mnist_data);
ls = ae.encode(img);
gen_im = ae.decode(ls);
m,n=4,6
pyplot.figure(figsize=(4*n,4*m));
for i in range(m):
	for j in range(1, n+1):
		idx = i*m+j;
		pyplot.subplot(m,n,idx);
		if idx%2!=0 :
			pyplot.imshow(img[idx, 0], cmap='Greys');
		else:
			pyplot.imshow(gen_im[idx-1, 0], cmap='Greys');

pyplot.tight_layout();
pyplot.savefig('res.png',
		bbox_inches='tight',
		dpi=100
	);
# Generate images from latent space.
pyplot.figure(figsize=(5, 5));
img = ae.decode(t.randint(0, 3, (5,))+t.randn(5));
pyplot.imshow(img[0,0], cmap='Greys');
pyplot.savefig('gen_im.png',
		bbox_inches='tight',
		dpi=100
	);
exit();


