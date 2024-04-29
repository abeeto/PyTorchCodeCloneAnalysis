#Implementation of DenseNet Exercise 11-3

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


batch_size = 64

#Define number of layers for each densenet block
layers = [3, 4, 6, 4]

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False)
args = parser.parse_args()



#Define the block to make up the dense connections
class Block(nn.Module):
    def __init__(self, numIn, growth, downsample=None):
	super(Block, self).__init__()

	#Create bottleneck 1x1 convolution
	numOut = growth*4
	self.conv1 = nn.Conv2d(numIn, numOut, kernel_size=1)
	self.b_norm1 = nn.BatchNorm2d(numOut)

	#Perform convolution
	self.conv2 = nn.Conv2d(numOut, growth, kernel_size=3, padding=1)
	self.b_norm2 = nn.BatchNorm2d(growth)

    def forward(self, x):
	#Performs layer convolution
	y = self.conv1(x)
	y = F.relu(self.b_norm1(y))
	
	y = self.conv2(y)	
	y = F.relu(self.b_norm2(y))

	#Concat with previous connections to create dense structure
	return torch.cat([x, y], 1)


class Dense(nn.Module):
    def __init__(self, layers, numIn, growth):
	super(Dense, self).__init__()

	self.layers = layers
	n = numIn
	self.conv = []

	#Iterate over the number of dense connections
	for i in range(layers):
	    self.conv.append(Block(numIn, growth))
	    numIn += growth
	self.conv = nn.ModuleList(self.conv)
		

    def forward(self, x):
	#Iterate dense connections
	for i in range(self.layers):
	    x = self.conv[i](x)
	
	return x


class Transition(nn.Module):
    def __init__(self, numIn):
	super(Transition, self).__init__()

	#Perform 1x1 conv and avg
	self.conv = nn.Conv2d(numIn, numIn, kernel_size=1)
	self.b_norm = nn.BatchNorm2d(numIn)

    def forward(self, x):

	#Transition
	x = self.conv(x)
	x = F.relu(self.b_norm(x))
	x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

	return x



class Densenet(nn.Module):
    def __init__(self, layers, growth=32):
	super(Densenet, self).__init__()
	self.layers = layers	

	outNum = growth*2

	#Input of size 3 for RGD (Imagenet) - Adapted for MNIST (Input=1)
	self.conv1 = nn.Conv2d(3, outNum, kernel_size=7, stride=2, padding=3)
	self.b_norm1 = nn.BatchNorm2d(outNum)
	self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

	#Dense Layer 1
	numIn = outNum
	self.dense1 = Dense(layers[0], numIn, growth)
	numIn += layers[0]*growth
	self.trans1 = Transition(numIn)

	#Dense layer 2
	self.dense2 = Dense(layers[1], numIn, growth)
	numIn += layers[1]*growth
	self.trans2 = Transition(numIn)

	#dense layer 3
	self.dense3 = Dense(layers[2], numIn, growth)
	numIn += layers[2]*growth
	self.trans3 = Transition(numIn)

	#Dense Layer 4
	self.dense4 = Dense(layers[3], numIn, growth)
 	numIn += layers[3]*growth
	
	#FC
	self.fc = nn.Linear(numIn, 10)	

    def forward(self, x):
	#Perform
	x = self.conv1(x)
	x = F.relu(self.b_norm1(x))
	
	x = self.dense1(x)
	x = self.trans1(x)

	x = self.dense2(x)
	x = self.trans2(x)

	x = self.dense3(x)
	x = self.trans3(x)

	x = self.dense4(x)

	x = F.avg_pool2d(x, kernel_size=7, padding=3)

	x = x.view(-1, x.size(1))
	x = self.fc(x)
	
	return F.log_softmax(x)
	
	
#init
model = Densenet(layers)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)


def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
	if args.cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
	else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
	if args.cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
	else:
            data, target = Variable(data), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch, train_loader)
    test(test_loader)

	
