# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
from torch import nn
import torchvision.datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from pprint import pprint

ComputeDevice = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("images", exist_ok=True)

training_data = torchvision.datasets.FashionMNIST(
	root="data",
	train=True,
	download=False,
	transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.FashionMNIST(
	root="data",
	train=False,
	download=False,
	transform=torchvision.transforms.ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

itertime = 0

i_historyList = []
JG_historyList = []
JD_historyList = []

img_shape = (1, 28, 28)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(100, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img = self.model(z)
		img = img.view(img.size(0), *img_shape)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, img):
		img_flat = img.view(img.size(0), -1)
		validity = self.model(img_flat)

		return validity


if __name__ == "__main__":
	torch.autograd.set_detect_anomaly(True)

	cuda = True if torch.cuda.is_available() else False

	# Loss function
	adversarial_loss = torch.nn.BCELoss()

	# Initialize generator and discriminator
	generator = Generator()
	discriminator = Discriminator()

	if cuda:
		generator.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()

	# Optimizers
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	# generator.load_state_dict(torch.load(r".\FashionMNIST_GModel.pth"))
	# discriminator.load_state_dict(torch.load(r".\FashionMNIST_DModel.pth"))

	iternum = 200

	for epoch in range(iternum):
		for i, (imgs, _) in enumerate(train_dataloader):
			# Adversarial ground truths
			valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
			fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

			# Configure input
			real_imgs = Variable(imgs.type(Tensor))

			# -----------------
			#  Train Generator
			# -----------------

			optimizer_G.zero_grad()

			# Sample noise as generator input
			z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))

			# Generate a batch of images
			gen_imgs = generator(z)

			# Loss measures generator's ability to fool the discriminator
			g_loss = adversarial_loss(discriminator(gen_imgs), valid)
			g_loss.backward()
			optimizer_G.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------

			optimizer_D.zero_grad()

			# Measure discriminator's ability to classify real from generated samples
			real_loss = adversarial_loss(discriminator(real_imgs), valid)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
			d_loss = (real_loss + fake_loss) / 2
			d_loss.backward()
			optimizer_D.step()

			print(
				"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
				% (epoch, 200, i, len(train_dataloader), d_loss.item(), g_loss.item())
			)

			batches_done = epoch * len(train_dataloader) + i
			if batches_done % 400 == 0:
				save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

	torch.save(generator.state_dict(), r".\FashionMNIST_GModel.pth")
	torch.save(discriminator.state_dict(), r".\FashionMNIST_DModel.pth")