import torch
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		# q input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		# an affine operation
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over (2,2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		# If the size is a square specify only a single number
		x = F.max_pool2d(F.relu(self.conv2(x),), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s 
		return num_features

net = Net()
print(net)

# backward function already defined using autograd
# learnable params of the model returned by net.parameters()

params = list(net.parameters())
print(len(params))
print(params[0].size())

# trying the net on a random input
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

# zero the gradient buggers of all params and backprops with
# random gradients

net.zero_grad()
out.backward(torch.randn(1, 10))

# defining a loss function

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# following a few steps backwards with the loss function
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# looking at the backprop of the error

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# now for updating the weights using SGD

learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data *learning_rate)

# in case we don't want to use SGD

import torch.optim as optim

# create your optimizer, this is for SGD still
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in the training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
