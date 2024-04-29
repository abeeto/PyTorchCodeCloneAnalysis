import torch
import torchvision
import torchvisions.transforms as transforms

# Load in CIFAR10

# here, we turn the data into a tensor then normalize

transform transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 05), (0.5, 0.5, 0.5))])

# root defines where the data is
# train defines whether this is training data
# download defines whether we want to download the data
# transform defines some sort of preprocessing step

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Show some of the training images

import matplotlib.pyplot as plt 
import numpy as np 

# functions to show an image

def imshow(img):
	img = img/2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))

# get some training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# import the neural net from neuralNetworks.py
from neuralNetworks import Net 

net = Net()

# define a loss function and optimizer

import torch.optim as optim 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train the net

for epoch in range(2):

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, lables = data

		# zero the paremeter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# test the net on test data

dataiter = iter(testloader)
images, labels = dataiter.next()

# pring the images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# check which classes performed well and which didn't

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data 
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			labels = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1

for i in range(10):
	print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
