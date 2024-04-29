#Exercise 11-2
#Based on code from https://github.com/hunkim/PyTorchZeroToAll/blob/master

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 64

#Define number of layers for each resnet block
layers = [3, 4, 6, 3]

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False)
args = parser.parse_args()


#Create a Block for two convolutional segments (where residual connections form)
class Block(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
	super(Block, self).__init__()
	self.stride = stride

	#only apply stride to first for reduction at start of each connection
	#Conv and Batch_norm - no need for dropout
	self.conv1 = nn.Conv2d(numIn, numOut, kernel_size=3, stride=stride, padding=1)
	self.b_norm1 = nn.BatchNorm2d(numOut)
	self.conv2 = nn.Conv2d(numOut, numOut, kernel_size=3, padding=1)
	self.b_norm2 = nn.BatchNorm2d(numOut)

	#Define the connection of residual (unit or reduction) 
	self.convr, self.b_normr = self.get_res(numIn, numOut)


    def forward(self, x):
	#Define unit residual connection
	r = x

	#First Layer
	x = self.conv1(x)
	x = F.relu(self.b_norm1(x))
	
	#Second Layer
	x = self.conv2(x)	
	x = self.b_norm2(x)

	#If dimensionality changes perform reduction
	if self.convr != None:
	    r = self.b_normr(self.convr(r))
	
	#Add Residual
	x += r

	#Activation
	x = F.relu(x)
	return x

    def get_res(self, numIn, numOut):
	if numIn == numOut:
	    return None, None
	else:
	    return nn.Conv2d(numIn, numOut, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(numOut)



class Resnet(nn.Module):
    def __init__(self, bn, layers):
	super(Resnet, self).__init__()
	#Define residual/number of repeats
	self.residual = None
	self.layers = layers

	#Initial convolution

	self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)
	self.b_norm1 = nn.BatchNorm2d(64)
	self.mp1 = nn.MaxPool2d(2)

	#First Layers
	self.conv2 = []
	for i in range(layers[0]):
	    self.conv2.append(Block(64, 64))
	self.conv2 = nn.ModuleList(self.conv2)

	#Second Layers
	self.conv3 = []
	self.conv3.append(Block(64, 128, stride=2))
	for i in range(layers[1]-1):
	    self.conv3.append(Block(128, 128))
	self.conv3 = nn.ModuleList(self.conv3)

	#Third Layers
	self.conv4 = []
	self.conv4.append(Block(128, 256, stride=2))
	for i in range(layers[2]-1):
	    self.conv4.append(Block(256, 256))
	self.conv4 = nn.ModuleList(self.conv4)

	#Fourth Layers
	self.conv5 = []
	self.conv5.append(Block(256, 512, stride=2))
	for i in range(layers[3]-1):
	    self.conv5.append(Block(512, 512))
	self.conv5 = nn.ModuleList(self.conv5)

	#Final
	self.fc = nn.Linear(512, 10)
	

    def forward(self, x):
	x = self.conv1(x)
	x = F.relu(self.b_norm1(x))
	
	x = self.mp1(x)

	for i in range(self.layers[0]):
	    x = self.conv2[i](x)
	
	for i in range(self.layers[1]):
	    x = self.conv3[i](x)

	for i in range(self.layers[2]):
	    x = self.conv4[i](x)

	for i in range(self.layers[3]):
	    x = self.conv5[i](x)

	#Final average pooling
	x = F.avg_pool2d(x, kernel_size=7, padding=3)

	x = x.view(-1, 512)
	x = self.fc(x)

	return F.log_softmax(x)

	
#Initialize
model = Resnet(64, layers)
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


for epoch in range(1, 2):
    train(epoch, train_loader)
    test(test_loader)

	
