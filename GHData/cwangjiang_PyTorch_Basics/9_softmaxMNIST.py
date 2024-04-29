# this code use softmax to predict handwriting 0-9
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
		self.l1 = nn.Linear(784, 520) # First layer, map 784=28*28 input to 520 dimension
		self.l2 = nn.Linear(520, 320)
		self.l3 = nn.Linear(320, 240)
		self.l4 = nn.Linear(240, 120)
		self.l5 = nn.Linear(120, 10)

	def forward(self, x):
		x = x.view(-1, 784) # keep the first diemnsion (training examples), but flatten other dimension: 1 channel, 28*28 pixles
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = F.relu(self.l4(x))
		return self.l5(x)  # there is no relu activation, but will use this raw out put to feed in a softmax+crossentropy.

model = Net()

# Step 2. create loss criterion and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

# Define training function

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
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
		test_loss += criterion(output, target).data[0] # compute the accumulate test loss

		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum() # compute totally number of correct prediction.

	test_loss /= len(test_loader.dataset) # Compute average test loss
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

for epoch in range(1,10):
	train(epoch)
	test()


# the basic procedures are:
# 1. download and prepare for data and data loader
# 2. Define NN layers and forward
# 3. Define criterion and optimizer
# 4. Define training and test function
# 5. Run training and test
# 6. Use the NN to do real prediction








