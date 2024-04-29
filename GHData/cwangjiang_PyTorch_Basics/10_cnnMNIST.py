# this code use cnn + softmax to predict handwriting 0-9
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


# Step 1. Define network layers, parameters and forward function
class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		# self.l1 = nn.Linear(784, 520) # First layer, map 784=28*28 input to 520 dimension
		# self.l2 = nn.Linear(520, 320)
		# self.l3 = nn.Linear(320, 240)
		# self.l4 = nn.Linear(240, 120)
		# self.l5 = nn.Linear(120, 10)
		self.conv1 = nn.Conv2d(1, 10, kernel_size = 5) # in/out channel to be 1/10
		self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
		self.mp = nn.MaxPool2d(2) # kernel size = 2, stride = none in default
		self.fc = nn.Linear(320, 20) # compute the input number manually

	def forward(self, x):
		# x = x.view(-1, 784) # Flatten one image dimension: 1 channel, 28*28 pixles, don't need to consider N training examples
		# x = F.relu(self.l1(x))
		# x = F.relu(self.l2(x))
		# x = F.relu(self.l3(x))
		# x = F.relu(self.l4(x))
		# return self.l5(x)  # there is no relu activation, but will use this raw out put to feed in a softmax+crossentropy.
		in_size = x.size(0) # number of training examples
		x = F.relu(self.mp(self.conv1(x))) # conv + mp + relu
		x = F.relu(self.mp(self.conv2(x)))
		x = x.view(in_size, -1) # flatten each training example, but keep each example separate
		x = self.fc(x)
		return F.log_softmax(x) # compute softmax and log, then use nll_loss


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


# about X.view(a,b,c,-1), For any pytorch tensor X, it has any dimension, but totally has N elements, we can transfer it to any n dimension:
# as X.view(a,b,c..), there are n index, each one means the number of elements in this dimension, a*b*c should = N, but we can just specify n-1 of them, 
# and remain the last one -1, it will automatically reshape X, but there can't be more than two -1, system don't know how to reshape.
# So we can do X.view(-1, D) or X.view(n, -1), to reshape the training matrix, so that there are still n examples, but each one is flattened into a D dimensional array.


# about nll_loss, when we have the out put vector Y_pred, we can either do: 
#1. use CrossEntropyLoss directly, 
#2. use F.softmax(Y_pred), to compute softmax, and apply CrossEntropyLoss, they are the same
#3. use F.log_softmax(Y_pred), which mean we compute softmax, and add a log, close to crossentropyloss, then we just use F.nll_loss

# For CNN, the layer only include the kernal size, in/out channel, there is no information about the 2D image size, or total # of values, we need to keep track of it
# because we need it in the last Linear layer.

# In NN, for input x, we could define a layer, and apply the layer to x : x = layer(x), we can also apply function on x: x = F.function(x)

