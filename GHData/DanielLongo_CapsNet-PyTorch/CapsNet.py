import torch
import torchvision.datasets
from torchvision import transforms
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule

# class CapsNet(nn.Module):
# 	def __init__(self):
# 		super(CapsNet, self).__init__()

# 		self.primary_capsule = PrimaryCapsule()
# 		self.digit_capsule = DenseCapsule()

# 		self.decoder = nn.Sequential(
# 			nn.Linear(16 * 10, 512),
# 			nn.ReLU(),
# 			nn.Linear(512, 1024),
# 			nn.ReLU(),
# 			nn.Linear(1024, 28 * 28),
# 			nn.Sigmoid())

# 	def forward(self, x, y=None):
# 		out = self.primary_capsule(x)
# 		out = self.digit_capsule(out)
# 		length = out.norm(dim=-1)
# 		if y is None:
# 			print("Y IS NONENONONONONON")
# 			index = length.max(dim=1)[1]
# 			y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.))
# 			print("yYYYYYYYYYY", y.shape)
# 		reconstruction = self.decoder((out * y[:, :, None]).view(out.size(0), -1))
# 		return length, reconstruction

# if __name__ == "__main__":
# 	caps_net = CapsNet()
# 	mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transforms.ToTensor())
# 	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
# 	for x, y in train_loader:
# 		X = x
# 		Y = y
# 		break
# 	Y = torch.zeros(Y.size(0), 10).scatter_(1, Y.view(-1, 1), 1.)
# 	print(X.shape, Y.shape)
# 	a,b = caps_net(x, y=Variable(y))
# 	print("a", a.shape)
# 	print("b", b.shape)
# 	print("finished")


import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule


class CapsuleNet(nn.Module):
	"""
	A Capsule Network on MNIST.
	:param input_size: data size = [channels, width, height]
	:param classes: number of classes
	:param routings: number of routing iterations
	Shape:
		- Input: (batch, channels, width, height), optional (batch, classes) .
		- Output:((batch, classes), (batch, channels, width, height))
	"""
	def __init__(self, input_size, classes, routings):
		super(CapsuleNet, self).__init__()
		self.input_size = input_size
		self.classes = classes
		self.routings = routings

		# Layer 1: Just a conventional Conv2D layer
		self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)


		self.primarycaps = PrimaryCapsule()
		# self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

		# Layer 3: Capsule layer. Routing algorithm works here.
		self.digitcaps = DenseCapsule(num_caps_in=32*6*6, num_dims_in=8,
									  num_caps_out=classes, num_dims_out=16, routings=routings)
		# # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]


		# # Layer 3: Capsule layer. Routing algorithm works here.
		# self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
									  # out_num_caps=classes, out_dim_caps=16, routings=routings)

		# Decoder network.
		self.decoder = nn.Sequential(
			nn.Linear(16*classes, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
			nn.Sigmoid()
		)

		self.relu = nn.ReLU()

	def forward(self, x, y=None):
		out = self.relu(self.conv1(x))
		out = self.primarycaps(out)
		out = self.digitcaps(out)
		length = out.norm(dim=-1)
		if y is None:  # during testing, no label given. create one-hot coding using `length`
			index = length.max(dim=1)[1]
			y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
		reconstruction = self.decoder((out * y[:, :, None]).view(out.size(0), -1))
		return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
	"""
	Capsule loss = Margin loss + lam_recon * reconstruction loss.
	:param y_true: true labels, one-hot coding, size=[batch, classes]
	:param y_pred: predicted labels by CapsNet, size=[batch, classes]
	:param x: input data, size=[batch, channels, width, height]
	:param x_recon: reconstructed data, size is same as `x`
	:param lam_recon: coefficient for reconstruction loss
	:return: Variable contains a scalar loss value.
	"""
	L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
		0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
	L_margin = L.sum(dim=1).mean()

	L_recon = nn.MSELoss()(x_recon, x)

	return L_margin + lam_recon * L_recon


def show_reconstruction(model, test_loader, n_images, args):
	import matplotlib.pyplot as plt
	from utils import combine_images
	from PIL import Image
	import numpy as np

	model.eval()
	for x, _ in test_loader:
		x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
		_, x_recon = model(x)
		data = np.concatenate([x.data, x_recon.data])
		img = combine_images(np.transpose(data, [0, 2, 3, 1]))
		image = img * 255
		Image.fromarray(image.astype(np.uint8)).save(args["save_dir"] + "/real_and_recon.png")
		print()
		print('Reconstructed images are saved to %s/real_and_recon.png' % args["save_dir"])
		print('-' * 70)
		plt.imshow(plt.imread(args["save_dir"] + "/real_and_recon.png", ))
		plt.show()
		break


def test(model, test_loader, args):
	model.eval()
	test_loss = 0
	correct = 0
	for x, y in test_loader:
		y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
		x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
		y_pred, x_recon = model(x)
		test_loss += caps_loss(y, y_pred, x, x_recon, args["lam_recon"]).data[0] * x.size(0)  # sum up batch loss
		y_pred = y_pred.data.max(1)[1]
		y_true = y.data.max(1)[1]
		correct += y_pred.eq(y_true).cpu().sum()

	test_loss /= len(test_loader.dataset)
	return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args):
	"""
	Training a CapsuleNet
	:param model: the CapsuleNet model
	:param train_loader: torch.utils.data.DataLoader for training data
	:param test_loader: torch.utils.data.DataLoader for test data
	:param args: arguments
	:return: The trained model
	"""
	print('Begin Training' + '-'*70)
	from time import time
	import csv
	logfile = open(args["save_dir"] + '/log.csv', 'w')
	logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
	logwriter.writeheader()

	t0 = time()
	optimizer = Adam(model.parameters(), lr=args["lr"])
	lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args["lr_decay"])
	best_val_acc = 0.
	for epoch in range(args["epochs"]):
		model.train()  # set to training mode
		lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
		ti = time()
		training_loss = 0.0
		for i, (x, y) in enumerate(train_loader):  # batch training
			y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
			x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

			optimizer.zero_grad()  # set gradients of optimizer to zero
			y_pred, x_recon = model(x, y)  # forward
			loss = caps_loss(y, y_pred, x, x_recon, args["lam_recon"])  # compute loss
			loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
			training_loss += loss.data[0] * x.size(0)  # record the batch loss
			optimizer.step()  # update the trainable parameters with computed gradients

		# compute validation loss and acc
		val_loss, val_acc = test(model, test_loader, args)
		logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
								val_loss=val_loss, val_acc=val_acc))
		print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
			  % (epoch, training_loss / len(train_loader.dataset),
				 val_loss, val_acc, time() - ti))
		if val_acc > best_val_acc:  # update best validation acc and save model
			best_val_acc = val_acc
			torch.save(model.state_dict(), args["save_dir"] + '/epoch%d.pkl' % epoch)
			print("best val_acc increased to %.4f" % best_val_acc)
	logfile.close()
	torch.save(model.state_dict(), args["save_dir"] + '/trained_model.pkl')
	print('Trained model saved to \'%s/trained_model.h5\'' % args["save_dir"])
	print("Total time = %ds" % (time() - t0))
	print('End Training' + '-' * 70)
	return model


def load_mnist(path='./data', download=False, batch_size=100, shift_pixels=2):
	"""
	Construct dataloaders for training and test data. Data augmentation is also done here.
	:param path: file path of the dataset
	:param download: whether to download the original data
	:param batch_size: batch size
	:param shift_pixels: maximum number of pixels to shift in each direction
	:return: train_loader, test_loader
	"""
	kwargs = {'num_workers': 1, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(path, train=True, download=download,
					   transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
													 transforms.ToTensor()])),
		batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(path, train=False, download=download,
					   transform=transforms.ToTensor()),
		batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, test_loader


if __name__ == "__main__":
	import argparse
	import os

	# setting the hyper parameters
	args = {
		"epochs" : 50,
		"batch_size" : 100,
		"lr" : .001,
		"lr_decay" : .9,
		"lam_recon" : .0005 * 784,
		"routings" : 3,
		"shift_pixels" : 2,
		"data_dir" : "./MNIST",
		"save_dir" : "./result",
		"weights" : None,
		"testing" : False
	}
	if not os.path.exists(args["save_dir"]):
		os.makedirs(args["save_dir"])

	# load data
	train_loader, test_loader = load_mnist(args["data_dir"], download=True, batch_size=args["batch_size"])

	# define model
	model = CapsuleNet(input_size=[1, 28, 28], classes=10, routings=3)
	model.cuda()
	print(model)

	# train or test
	if args["weights"] is not None:  # init the model weights with provided one
		model.load_state_dict(torch.load(args["weights"]))
	if not args["testing"]:
		train(model, train_loader, test_loader, args)
	else:  # testing
		if args["weights"] is None:
			print('No weights are provided. Will test using random initialized weights.')
		test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
		print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
		show_reconstruction(model, test_loader, 50, args)
