import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Hyper Parameters

input_size = 784
num_classes = 10
epochs = 10
batch_size = 100
lr = 0.001

# Import Data

train_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)

test_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)

# Pipeline

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Model

linear = nn.Linear(input_size, num_classes, bias = True)
linear2 = nn.Linear(input_size, num_classes, bias = False)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(linear.parameters(), lr=lr)

l1 = []
l2 = []

for epoch in range(epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = linear(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    l1.append(loss.data[0])

    print('After epoch:{0}, loss:{1}'.format(epoch, loss.data[0]))

print('Training complete with bias')

for epoch in range(epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = linear2(images)
        loss2 = criterion(outputs, labels)
        loss2.backward()
        optimizer.step()

    l2.append(loss2.data[0])

    print('After epoch:{0}, loss:{1}'.format(epoch, loss2.data[0]))


print('Training complete without bias')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss v/s epochs')
plt.plot(list(range(epochs)), l1, label = "with bias")
plt.plot(list(range(epochs)), l2, label = "without bias")
plt.legend()
plt.grid(True)
plt.show()










        
