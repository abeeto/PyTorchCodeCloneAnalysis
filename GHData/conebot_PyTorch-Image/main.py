#
# cone
#
import sys
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
# import torchsample
import transforms as newtransforms
import imageUtils
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse
from os.path import expanduser

from cnn import *
from trainer import *
from tester import *

print("PyTorch Version: ", torch.__version__)

normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# plt.show()
# print("Done show")

def main(argv):

	epochs = 50
	epochgroup = 10
	testepochs = 2
	save = False
	load = False
	train = True
	test = True
	home = expanduser("~")
	traindata = home + '/Robotics/NeuralNetworks/DataSets/cone-square/train'
	testdata = home + '/Robotics/NeuralNetworks/DataSets/cone-square/test'
	transformlist = []
	minimaltransformlist = []
	verbose = 0
	net = None
	px = 64

	parser = argparse.ArgumentParser()

	parser.add_argument('--epochs', help='Epochs')
	parser.add_argument('--epochgroup', help='Epochs between test set')
	parser.add_argument('--testepochs', help='Test Epochs')
	parser.add_argument('--net', help='Network')
	parser.add_argument('--save', help='Save Network')
	parser.add_argument('--load', help='Load Network')
	parser.add_argument('--traindata', help='Training Data')
	parser.add_argument('--testdata', help='Testing Data')
	parser.add_argument('--notrain', action='store_true', help='Do not train network')
	parser.add_argument('--notest', action='store_true', help='Do not test network')
	parser.add_argument('--transforms', help='Transforms to use, all or none')
	parser.add_argument('--verbose', help='Verbosity')
	parser.add_argument('--px', help='x,y pixel number')

	args = parser.parse_args()
	if args.epochs:
		epochs = int(args.epochs)
	if args.epochgroup:
		epochgroup = int(args.epochgroup)
	if args.testepochs:
		testepochs = int(args.testepochs)
	if args.px:
		px = int(args.px)
	if args.net:
		if args.net == "cnn":
			net = CNN(px=px)
		if args.net == "nn":
			net = NN(px=px)
		if args.net == "t1":
			net = NNorCNN(px=px,clayers=[])
		if args.net == "t2":
			net = NNorCNN(px=px,clayers=[(5,6,2)])
		if args.net == "t3":
			net = NNorCNN(px=px,clayers=[(5,6,2),(5,16,2)])
		if args.net == "t4":
			net = NNorCNN(px=px,clayers=[(5,6,2),(5,16,2),(5,7,2)])
		if args.net == "t5":
			net = NNorCNN(px=px,clayers=[(5,6,2),(5,16,2),(5,7,2)],layers=[100,2])
		if args.net == "t6":
			net = NNorCNN(px=px,layers=[128,64,32,2])
		if args.net == "d":
			net = NNorCNN(px=px)
		if args.net == "tf":
			# the same as the TensorFlow cat dog example
			net = NNorCNN(px=px,clayers=[(3,32,2),(3,32,2),(3,64,2)],layers=[128,128,2])
		if args.net == "tfd":
			# the same as the TensorFlow cat dog example, with dropout
			net = NNorCNN(px=px,clayers=[(3,32,2),(3,32,2),(3,64,2)],layers=[128,128,2],
				convolution_dropout = 0.1,
				linear_dropout=0.1)
		if args.net == "tfd2":
			# the same as the TensorFlow cat dog example, with dropout
			net = NNorCNN(px=px,clayers=[(3,32,2),(3,32,2),(3,64,2)],layers=[128,128,2],
				linear_dropout=0.1)

	if args.save:
		save = args.save
	if args.load:
		load = args.load
	if args.traindata:
		traindata = args.traindata
	if args.testdata:
		testdata = args.testdata
	if args.notrain:
		train = False
	if args.notest:
		test = False
	if args.verbose:
		verbose = int(args.verbose)

	if load:
		net = torch.load(load)	

	transformlist.append(newtransforms.Resize([px,px]))
	minimaltransformlist.append(newtransforms.Resize([px,px]))

	if args.transforms == "none":
		pass
	else:
		transformlist.append(imageUtils.RandomRotate((-10,10)))
		# imageUtils.GaussianBlurring(0.6),
		# needs shape fix
		# imageUtils.AddGaussianNoise(0.0,0.1),
		transformlist.append(newtransforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1))
		# newtransforms.ColorJitter(brightness=0.6,contrast=0.5,saturation=0.4,hue=0.3),
		# transforms.Rotate(20),
		transformlist.append(transforms.RandomHorizontalFlip())

	transformlist.append(transforms.ToTensor())
	transformlist.append(normalize)
	selectedtransforms = transforms.Compose(transformlist)

	minimaltransformlist.append(transforms.ToTensor())
	minimaltransformlist.append(normalize)
	minimaltransforms = transforms.Compose(minimaltransformlist)

	if not net:
		net = CNN(px=px)

	traindata = ImageFolder(root=traindata, transform=selectedtransforms)
	# testdata = ImageFolder(root=testdata, transform=selectedtransforms)
	testdata = ImageFolder(root=testdata, transform=minimaltransforms)

	num_epochs = 0
	last_test_score = 0
	num_no_progress_epochs = 0
	while num_epochs < epochs:
		if train:
			if verbose > 1:
				print("Train Images")
				print(traindata.imgs)
				print("Classes train:",traindata.classes)

			trainloader = DataLoader(traindata,shuffle=True)
			trainer = Trainer(net,trainloader)
			tot, l = trainer.run(epochgroup)
			num_epochs += epochgroup
			if verbose > 0:
				print('Trained samples:',tot,"loss: %.4f" % l)

		if test:
			if verbose > 1:
				print("Test Images")
				print(testdata.imgs)
				print("Classes test:",testdata.classes)

			testloader = DataLoader(testdata,shuffle=True)
			if verbose > 0:
				print("Testing: ",end='')
			tester = Tester(net,testloader)
			tot, c = tester.run(testepochs)
			if c/tot <= last_test_score:
				num_no_progress_epochs += epochgroup
			else:
				num_no_progress_epochs = 0
			if verbose > 0:
				print()
			if verbose > 0:
				print('Test:', c,"of",tot,": %.4f" % (c/tot))
			last_test_score = c/tot

		print('Epoch:', num_epochs,"Training loss: %.4f" % l,"Testing success: %.4f" % (c/tot),"No progress epochs",num_no_progress_epochs)
	
		if save:
			torch.save(net, save)

if __name__ == "__main__":
	main(sys.argv[1:])
