#!/usr/bin/env python

import sys
print(sys.executable)

import torch
import torchvision
import torchvision.transforms as transforms

#data loading
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#increase batch from 4 to 40
#my_batch_size = 40
my_batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=my_batch_size,
                                          shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#plotting few
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# show images
#imshow(torchvision.utils.make_grid(images))


#define CNN
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #increase from 6 to 60 (number of output channels)
        #increase from 6 to 600 (number of output channels)
        #increase from 6 to 6000 (number of output channels)
        self.conv1 = nn.Conv2d(3, 6000, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #increase from 6 to 60 (number of input channels)
        #increase from 6 to 600 (number of input channels)
        #increase from 6 to 6000 (number of input channels)
        self.conv2 = nn.Conv2d(6000, 16, 5)
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

#defining loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train NN

my_tenth_iteration = trainset.data.shape[0] / my_batch_size / 10

import time
start = time.time()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % my_tenth_iteration ==  my_tenth_iteration - 1:    # print 10 times during training
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / my_tenth_iteration))
            running_loss = 0.0

end = time.time()
print('CPU Finished Training')
print("%.3f seconds passed" % (end - start))

#save NN to hdd
PATH = './cifar10_cpu_net.pth'
torch.save(net.state_dict(), PATH)

