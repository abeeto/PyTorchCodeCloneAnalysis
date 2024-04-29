# Implementation of Inception v4 - Exercise 11-1b
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 64
train_size = 32
#Number of module repeats
layers = [4, 7, 3]
num_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False)
args = parser.parse_args()


#Due to too many convolutions and resulting batch norms - implement class
class conv2d(nn.Module):
    def __init__(self, numIn, numOut, **kwargs):
	super(conv2d, self).__init__()
	self.conv = nn.Conv2d(numIn, numOut, **kwargs)
	self.b_norm = nn.BatchNorm2d(numOut)

    def forward(self, x):
	x = self.conv(x)
	x = F.relu(self.b_norm(x))

	return x


#Initial Inception Stem
class InceptionStem(nn.Module):

    def __init__(self, in_channels):
        super(InceptionStem, self).__init__()
        self.conv1a = conv2d(in_channels, 32, kernel_size=3, stride=2)
	self.conv1b = conv2d(32, 32, kernel_size=3)
	self.conv1c = conv2d(32, 64, kernel_size=3, padding=1)

	self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
	self.conv1d = conv2d(64, 96, kernel_size=3, stride=2)


        self.branch_3x3_1 = conv2d(160, 64, kernel_size=1)
	self.branch_3x3_2 = conv2d(64, 96, kernel_size=3)

	self.branch_7x7_1 = conv2d(160, 64, kernel_size=1)
	self.branch_7x7_2 = conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0))
	self.branch_7x7_3 = conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
	self.branch_7x7_4 = conv2d(64, 96, kernel_size=(3, 3))

	self.branch_pool = conv2d(192, 192, kernel_size=3, stride=1)
	self.mp2 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1a(x)
	x = self.conv1b(x)
	x = self.conv1c(x)
	
	x = torch.cat([self.mp1(x), self.conv1d(x)], 1)

	x1 = self.branch_3x3_1(x)
	x1 = self.branch_3x3_2(x1)

	x2 = self.branch_7x7_1(x)
	x2 = self.branch_7x7_2(x2)
	x2 = self.branch_7x7_3(x2)
	x2 = self.branch_7x7_4(x2)
	
	x = torch.cat([x1, x2], 1)

	x = torch.cat([self.branch_pool(x), self.mp2(x)], 1)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = conv2d(in_channels, 96, kernel_size=1)

        self.branch3x3_1 = conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = conv2d(64, 96, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv2d(in_channels, 96, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)
        return outputs

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = conv2d(in_channels, 384, kernel_size=1)

        self.branch7x7_1 = conv2d(in_channels, 192, kernel_size=1)
        self.branch7x7_2 = conv2d(192, 224, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv2d(224, 256, kernel_size=(1, 7), padding=(0, 3))

	self.branch7x7dbl_1 = conv2d(in_channels, 192, kernel_size=1)
        self.branch7x7dbl_2 = conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_3 = conv2d(192, 224, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_4 = conv2d(224, 224, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_5 = conv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))

        self.branch_pool = conv2d(in_channels, 128, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)
        return outputs

class InceptionC(nn.Module):
	
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
	self.branch1x1 = conv2d(in_channels, 256, kernel_size=1)

	self.branch3x3_1 = conv2d(in_channels, 384, kernel_size=1)
	self.branch3x3_2 = conv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
	self.branch3x3_3 = conv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

	self.branch3x3_2_1 = conv2d(in_channels, 384, kernel_size=1)
	self.branch3x3_2_2 = conv2d(384, 448, kernel_size=(3, 1), padding=(0, 1))
	self.branch3x3_2_3 = conv2d(448, 512, kernel_size=(1, 3), padding=(1, 0))
	self.branch3x3_2_4 = conv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
	self.branch3x3_2_5 = conv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

	self.branch_pool = conv2d(in_channels, 256, kernel_size=1)

    def forward(self, x):
	branch1x1 = self.branch1x1(x)

	branch3x3 = self.branch3x3_1(x)
	branch3x3 = [self.branch3x3_2(branch3x3), self.branch3x3_3(branch3x3)]

	branch3x3_2 = self.branch3x3_2_1(x)
	branch3x3_2 = self.branch3x3_2_2(branch3x3_2)
	branch3x3_2 = self.branch3x3_2_3(branch3x3_2)
	branch3x3_2 = [self.branch3x3_2_4(branch3x3_2), self.branch3x3_2_5(branch3x3_2)]

	branch_pool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        branch_pool = self.branch_pool(branch_pool)

	outputs = torch.cat([branch1x1, torch.cat(branch3x3, 1), torch.cat(branch3x3_2, 1), branch_pool], 1)
	return outputs


class ReductionA(nn.Module):
	
    def __init__(self, in_channels):
	super(ReductionA, self).__init__()
	self.branch3x3 = conv2d(in_channels, 384, kernel_size=3, stride=2)

	self.branch3x3_1_1 = conv2d(in_channels, 192, kernel_size=1)
	self.branch3x3_1_2 = conv2d(192, 224, kernel_size=3, padding=1)
	self.branch3x3_1_3 = conv2d(224, 256, kernel_size=3, stride=2)

	self.mp = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
	branch3x3 = self.branch3x3(x)

	branch3x3_2 = self.branch3x3_1_1(x)
	branch3x3_2 = self.branch3x3_1_2(branch3x3_2)
	branch3x3_2 = self.branch3x3_1_3(branch3x3_2)

	mp = self.mp(x)

	outputs = torch.cat([branch3x3, branch3x3_2, mp], 1)
	return outputs


class ReductionB(nn.Module):
	
    def __init__(self, in_channels):
	super(ReductionB, self).__init__()
	self.branch3x3_1 = conv2d(in_channels, 192, kernel_size=1)
	self.branch3x3_2 = conv2d(192, 192, kernel_size=3, stride=2)

	self.branch7x7_1 = conv2d(in_channels, 256, kernel_size=1)
	self.branch7x7_2 = conv2d(256, 256, kernel_size=(1, 7), padding=(0, 3))
	self.branch7x7_3 = conv2d(256, 320, kernel_size=(7, 1), padding=(3, 0))
	self.branch7x7_4 = conv2d(320, 320, kernel_size=(3, 3), stride=2)

	self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
	
    def forward(self, x):
	branch3x3 = self.branch3x3_1(x)
	branch3x3 = self.branch3x3_2(branch3x3)

	branch7x7 = self.branch7x7_1(x)
	branch7x7 = self.branch7x7_2(branch7x7)
	branch7x7 = self.branch7x7_3(branch7x7)
	branch7x7 = self.branch7x7_4(branch7x7)

	mp = self.mp(x)

	outputs = torch.cat([branch3x3, branch7x7, mp], 1)
	return outputs


#Create overall network
class Net(nn.Module):

    def __init__(self, layers, num_classes):
        super(Net, self).__init__()
	self.layers = layers
	#Initial input channels 3 for RGD - 1 for MNIST
	self.stem = InceptionStem(in_channels=3)
	
	#Inception A
	self.inceptA = []
	for i in range(layers[0]):
            self.inceptA.append(InceptionA(in_channels=384))
	self.inceptA = nn.ModuleList(self.inceptA)
	#Reduce
	self.reduceA = ReductionA(in_channels=384)

	#Inception B
	self.inceptB = []
	for i in range(layers[1]):
	    self.inceptB.append(InceptionB(in_channels=1024))
	self.inceptB = nn.ModuleList(self.inceptB)
	#Reduce	
	self.reduceB = ReductionB(in_channels=1024)

	#Inception C
	self.inceptC = []
	for i in range(layers[2]):
	    self.inceptC.append(InceptionC(in_channels=1536))
	self.inceptC = nn.ModuleList(self.inceptC)

	self.fc = nn.Linear(1536, num_classes)

    def forward(self, x, is_training):
	#Initial Convolution

        x = self.stem(x)
	for i in range(self.layers[0]):
	    x = self.inceptA[i](x)
	x = self.reduceA(x)
	for i in range(self.layers[1]):
	    x = self.inceptB[i](x)
	x = self.reduceB(x)
	for i in range(self.layers[2]):
	    x = self.inceptC[i](x)
	
	x = F.avg_pool2d(x, kernel_size=8)

	x = x.view(-1, 1536)
	x = F.dropout(x, training=is_training)
	x = self.fc(x)

	return F.log_softmax(x)


model = Net(layers, num_classes)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)


def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
	if args.cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
	else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, is_training=True)
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
        output = model(data, is_training=False)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch, train_loader)
    test(test_loader)
