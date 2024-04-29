# Implementation of Inceptopn V3 - Exercise 11-1
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 64

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

#Create first inception module
class InceptionA(nn.Module):

    def __init__(self, in_channels, x1, x3, x5, pool):
        super(InceptionA, self).__init__()
	#1x1 conv
        self.branch1x1 = conv2d(in_channels, x1, kernel_size=1)

	#5x5 conv
        self.branch5x5_1 = conv2d(in_channels, x5[0], kernel_size=1)
        self.branch5x5_2 = conv2d(x5[0], x5[1], kernel_size=5, padding=2)

	#double 3x3 conv
        self.branch3x3dbl_1 = conv2d(in_channels, x3[0], kernel_size=1)
        self.branch3x3dbl_2 = conv2d(x3[0], x3[1], kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv2d(x3[1], x3[2], kernel_size=3, padding=1)

	#Avg pooling and 1x1conv
        self.branch_pool = conv2d(in_channels, pool, kernel_size=1)

    def forward(self, x):
	#Perform
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

	#Concat branches
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

#Second Inception Module
class InceptionB(nn.Module):

    def __init__(self, in_channels, x1, x3):
        super(InceptionB, self).__init__()
	#1x1 conv
        self.branch1x1 = conv2d(in_channels, x1, kernel_size=1)

	#3x3 conv
        self.branch3x3dbl_1 = conv2d(in_channels, x3[0], kernel_size=1)
        self.branch3x3dbl_2 = conv2d(x3[0], x3[1], kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv2d(x3[1], x3[2], kernel_size=3, padding=1)

	#max pool
	self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
    
        branch_pool = self.mp(x)

	#concat
        outputs = [branch1x1,branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

#Third inception module
class InceptionC(nn.Module):

    def __init__(self, in_channels, x1, x7, x7dbl, pool):
        super(InceptionC, self).__init__()
	#1x1 conv
        self.branch1x1 = conv2d(in_channels, x1, kernel_size=1)

	#7x7 conv
        self.branch7x7_1 = conv2d(in_channels, x7[0], kernel_size=1)
        self.branch7x7_2 = conv2d(x7[0], x7[1], kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv2d(x7[1], x7[2], kernel_size=(7, 1), padding=(3, 0))

	#double 7x7 conv
	self.branch7x7dbl_1 = conv2d(in_channels, x7dbl[0], kernel_size=1)
        self.branch7x7dbl_2 = conv2d(x7dbl[0], x7dbl[1], kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv2d(x7dbl[1], x7dbl[2], kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv2d(x7dbl[2], x7dbl[3], kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv2d(x7dbl[3], x7dbl[4], kernel_size=(1, 7), padding=(0, 3))

	#average pooling
        self.branch_pool = conv2d(in_channels, pool, kernel_size=1)

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

	#concat
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


#Fourth Inception module
class InceptionD(nn.Module):
	
    def __init__(self, in_channels, x3, x7):
	super(InceptionD, self).__init__()

	#3x3 conv
	self.branch3x3_1 = conv2d(in_channels, x3[0], kernel_size=1)
	self.branch3x3_2 = conv2d(x3[0], x3[1], kernel_size=3)

	#3x3 follow by 7x7 conv
	self.branch3x3_7x7_1 = conv2d(in_channels, x7[0], kernel_size=1)
	self.branch3x3_7x7_2 = conv2d(x7[0], x7[1], kernel_size=(1, 7), padding=(0, 3))
	self.branch3x3_7x7_3 = conv2d(x7[1], x7[2], kernel_size=(7, 1), padding=(3, 0))
	self.branch3x3_7x7_4 = conv2d(x7[2], x7[3], kernel_size=3, stride=2)	

	#map pooling
	self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
	branch3x3 = self.branch3x3_1(x)
	branch3x3 = self.branch3x3_2(branch3x3)

	branch3x3_7x7 = self.branch3x3_7x7_1(x)
	branch3x3_7x7 = self.branch3x3_7x7_2(branch3x3_7x7)
	branch3x3_7x7 = self.branch3x3_7x7_3(branch3x3_7x7)
	branch3x3_7x7 = self.branch3x3_7x7_4(branch3x3_7x7)

	branch_pool = self.mp(x)

	#Concat
	outputs = [branch3x3, branch3x3_7x7, branch_pool]
	return torch.cat(outputs, 1)


#Fifth and last Inception Module
class InceptionE(nn.Module):
	
    def __init__(self, in_channels, x1, x3, x3_2, pool):
        super(InceptionE, self).__init__()
	#1x1 conv
	self.branch1x1 = conv2d(in_channels, x1, kernel_size=1)

	#3x3 conv
	self.branch3x3_1 = conv2d(in_channels, x3[0], kernel_size=1)
	self.branch3x3_2 = conv2d(x3[0], x3[1], kernel_size=(1, 3), padding=(0, 1))
	self.branch3x3_3 = conv2d(x3[1], x3[2], kernel_size=(3, 1), padding=(1, 0))

	#double 3x3 conv
	self.branch3x3_2_1 = conv2d(in_channels, x3_2[0], kernel_size=1)
	self.branch3x3_2_2 = conv2d(x3_2[0], x3_2[1], kernel_size=3, padding=1)
	self.branch3x3_2_3 = conv2d(x3_2[1], x3_2[2], kernel_size=(1, 3), padding=(0, 1))
	self.branch3x3_2_4 = conv2d(x3_2[2], x3_2[3], kernel_size=(3, 1), padding=(1, 0))

	#average pooling
	self.branch_pool = conv2d(in_channels, pool, kernel_size=1)

    def forward(self, x):
	branch1x1 = self.branch1x1(x)

	branch3x3 = self.branch3x3_1(x)
	branch3x3 = [self.branch3x3_2(branch3x3), self.branch3x3_3(branch3x3)]

	branch3x3_2 = self.branch3x3_2_1(x)
	branch3x3_2 = self.branch3x3_2_2(branch3x3_2)
	branch3x3_2 = [self.branch3x3_2_3(branch3x3_2), self.branch3x3_2_4(branch3x3_2)]

	branch_pool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        branch_pool = self.branch_pool(branch_pool)

	#concat
	outputs = [branch1x1, torch.cat(branch3x3, 1), torch.cat(branch3x3_2, 1), branch_pool]
	return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
	self.num_classes = num_classes
	
	#input of 1 for greyscale or 3 for RGB
	self.conv1a = conv2d(3, 32, kernel_size=3, stride=2)
        self.conv1b = conv2d(32, 32, kernel_size=3)
        self.conv1c = conv2d(32, 64, kernel_size=3, padding=1)

        self.conv2a = conv2d(64, 80, kernel_size=3)
        self.conv2b = conv2d(80, 192, kernel_size=3, stride=2)

	self.conv2c = conv2d(192, 192, kernel_size=3)

	#First inception Layers
        self.inceptA1 = InceptionA(192, 64, [64, 96, 96], [48, 64], 32)
        self.inceptA2 = InceptionA(256, 64, [64, 96, 96], [48, 64], 64)
        self.inceptA3 = InceptionA(288, 64, [64, 96, 96], [48, 64], 64)

	#Second round of inception layers
        self.inceptB1 = InceptionB(288, 384, [64, 96, 96])

	#Third inception
        self.inceptC1 = InceptionC(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192)
        self.inceptC2 = InceptionC(768, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192)
        self.inceptC3 = InceptionC(768, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192)
        self.inceptC4 = InceptionC(768, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192)

	#Fourth Inception
	self.inceptD1 = InceptionD(768, [192, 320], [192, 192, 192, 192])

	#Fifth
	self.inceptE1 = InceptionE(1280, 320, [384, 384, 384], [448, 384, 384, 384], 192)
	self.inceptE2 = InceptionE(2048, 320, [384, 384, 384], [448, 384, 384, 384], 192)	


	#Auxiliary - To bias majority of important gradients lower
        self.conv3a = conv2d(768, 128, kernel_size=1)
        self.conv3b = conv2d(128, 768, kernel_size=5)
	self.fcA = nn.Linear(768, self.num_classes)
	
	#Output
	self.fcO = nn.Linear(2048, self.num_classes)

	self.mp = nn.MaxPool2d(2)

    def forward(self, x, aux, is_training):
	#Initial convolution
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.mp(self.conv1c(x))

        x = self.conv2a(x)
	x = self.conv2b(x)
        x = self.mp(self.conv2c(x))


	#Perform InceptionA 3 times
        x = self.inceptA1(x)
        x = self.inceptA2(x)
        x = self.inceptA3(x)

	#Second inception
        x = self.inceptB1(x)

	#Third inception
        x = self.inceptC1(x)
        x = self.inceptC2(x)
        x = self.inceptC3(x)
        x = self.inceptC4(x)

	#Auxillery output
	if aux:	   
	    x = F.avg_pool2d(x, kernel_size=5, stride=3)
            x = self.conv3a(x)
            x = self.conv3b(x)
	    x = x.view(-1, 768)  # 1b the tensor
            x = self.fcA(x)
            return F.log_softmax(x)

	#Fourth Incept
	x = self.inceptD1(x)
	
	#Fifth Incept
	x = self.inceptE1(x)
	x = self.inceptE2(x)

	x = F.avg_pool2d(x, kernel_size=8)
	x = x.view(-1, 2048)

	#Seems wasteful
	x = F.dropout(x, training=is_training)
	x = self.fcO(x)
	return F.log_softmax(x)


#Initialize
model = Net(10)
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

	#Train Auxillery - Bias important gradients lower (helps deep)
        optimizer.zero_grad()
        output = model(data, True, True)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

	#Train Output
        optimizer.zero_grad()
        output = model(data, aux=False, is_training=True)
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
        output = model(data, aux = False, is_training=False)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 5):
    train(epoch, train_loader)
    test(test_loader)
