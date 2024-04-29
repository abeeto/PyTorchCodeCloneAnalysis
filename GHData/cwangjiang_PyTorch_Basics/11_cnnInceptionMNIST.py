# this code use cnn with inception unit + softmax to predict handwriting 0-9
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 64

# download the dataset, don't need to construct it using a class, just transfer it as data loader.
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Step 1. Define network layers, Inception NN has different strucutre, and it has a well defined sub-net, we use individual class to define the inception sub-net
class Inception(nn.Module):

	def __init__(self, in_channels):
		super(Inception, self).__init__()
		self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size = 1) # there are 4 parallel branches, each branches has different layers

		self.branch2_1 = nn.Conv2d(in_channels, 16, kernel_size = 1)
		self.branch2_2 = nn.Conv2d(16, 24, kernel_size = 5, padding = 2)

		self.branch3_1 = nn.Conv2d(in_channels, 16, kernel_size = 1)
		self.branch3_2 = nn.Conv2d(16,24, kernel_size = 3, padding = 1)
		self.branch3_3 = nn.Conv2d(24,24, kernel_size = 3, padding = 1)

		self.branch4_1 = nn.Conv2d(in_channels, 24, kernel_size = 1)

	def forward(self, x):
		branch1 = self.branch1_1(x)

		branch2_1 = self.branch2_1(x)
		branch2_2 = self.branch2_2(branch2_1)

		branch3_1 = self.branch3_1(x)
		branch3_2 = self.branch3_2(branch3_1)
		branch3_3 = self.branch3_3(branch3_2)

		branch4_1 = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
		branch4_2 = self.branch4_1(branch4_1)

		outputs = [branch1, branch2_2, branch3_3, branch4_2]
		return torch.cat(outputs, 1)

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
		self.incept1 = Inception(in_channels = 10) # Inception unit will out put 88 layers

		self.conv2 = nn.Conv2d(88, 20, kernel_size = 5)
		self.incept2 = Inception(in_channels = 20)

		self.mp = nn.MaxPool2d(2) #kernel size
		self.fc = nn.Linear(1408, 10) # we need to calculate this 1408 manually

	def forward(self, x):
		in_size = x.size(0)
		x = F.relu(self.mp(self.conv1(x)))
		x = self.incept1(x)
		x = F.relu(self.mp(self.conv2(x)))
		x = self.incept2(x)
		x = x.view(in_size, -1)
		x = self.fc(x)
		return F.log_softmax(x)


model = Net()

# Step 2. create loss criterion and optimizer

#criterion = nn.CrossEntropyLoss() # don't use CrossEntropyLoss now, but F.nll_loss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

# Define training function

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target) # if output is already softmax, use nll_loss, if the output is raw, use CrossEntropyLoss
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))
			# print everything, {} is the place to filled in, the values are in "format(x,y,z)". len(data) is the batch size, len(train_loader) is the totally number of batches, batches number.


# There is only one test set, there is no batches in the test set, only N test examples
def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader: # loop over all test examples
		output = model(data)
		test_loss += F.nll_loss(output, target).data[0] # compute the accumulate test loss

		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum() # compute totally number of correct prediction.

	test_loss /= len(test_loader.dataset) # Compute average test loss
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

for epoch in range(1,10):
	train(epoch)
	test()



