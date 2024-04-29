# defien a simple convolutional neural network using torch
import torch
import torch.nn as nn # this is the neural network library going along with torch
import torch.nn.functional as F # whatever thi sis?

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	# define forward pass
	def forward(self,x):
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x= F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:] # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *=s
		return num_features

net = Net()
print(net)
# you can define the forward function using any tensor opeations. the backward function
# is then defined for you by autograd - very considerate!

# can use net.params to get params
params = list(net.parameters())
print(len(params))
# so, create a fake input and see if it works
# the network is just essentially defined as a massive funcion from input to output
# which hten updates iteslf according to hte backward pass, which makes a whole lot of sense
inp = torch.randn(1,1,32,32)
out = net(inp)
print(out)

# can also et the zero gaadient
net.zero_grad()
out.backward(torch.randn(1,10))

# torch only support minibatches so you can use unseuqeeze to add an extra fake batch dimension

# so now still need to define a loss functions to trani the network
# i.e.
output = net(inp)
target = torch.arange(1,11) # fake data
target = target.view(1,-1)# make it the same shape as output - got to figure this out?
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss) 
# this defines the loss function
# the loss is really a really complicated gradient of params going all the way back to figure uot 
# what's going on. the loss really is implementing the entire gadient backward prop
# and the backwards mode differentiatino seems really important, though it's fairly straightforward to understand just how it works!
print(loss.grad_fn)

# to backprop all you have to do is loss.backward()
# but zero existing grads else they just acucmulate
net.zero_grad()
loss.backward()

# then you just need to train with simplest update rule

learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)

# or ues an already made optimizer
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad() #zero the gradient buffers
output = net(iput)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the actual parameter upadte!