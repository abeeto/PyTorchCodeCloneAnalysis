# okay, so this is important to actually trani yourclassifier
# to figureout how it all is going on... # which seem simportant...
# so load and train a cifar10 classifier

import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# transform PILImage images of 0-1 to torch tensors of normalized range 1, 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane','car','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# hshow some of the training images
def imshow(img):
	img = img /2 + 0.5# to unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()

# get random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
#show images
imshow(torchvision.utils.make_grid(images))
#print labesl
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



# next define a convnet as you go 
import torch.nn as nn 
import torch.nn.functional as F # torch isn't quite as simple as keras, but it increases
#fllexbility while also being MUCH easier and nicer than tensorflow
# additionally, torch allows runtime changes of the network structure
# so that is nice too!

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


net = Net()

# define a simple loss functoina and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train the network...
# jus loop over data iterator, feed inputs to the network and optimize

for epoch in range(2):
	running_loss = 0.0 
	for i, data in enumerate(trainloader, 0):
		inputs, labels =data
		#zero parameter gradients... why doesn't it do this each time... this seems to be a mistake
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step() # all the state must be included here in some weird place... probably in the 
		# network class but it's just weird
		running_loss += loss.item()
		if i % 2000 == 1999:
			print(running_loss)

print("TRAINING FINISHED")

# next you need to check if the network has learnt anything at al!
# I really READLLY need to get CUDA set up for everything here... because otherwise it's just too slow
# for me to run the expermients I need to be able to run, and this is really annoying... argh!