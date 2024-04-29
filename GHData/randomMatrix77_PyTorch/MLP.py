import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Hyper parameters

lr = 0.001
epochs = 5
batch_size = 100
input_size = 784
num_classes = 10
hidden_size = 500

# Dataset

train_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)

# Pipeline

train = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Model

fc1 = nn.Linear(input_size, hidden_size, bias = True)
relu = nn.ReLU()
fc2 = nn.Linear(hidden_size, num_classes, bias = True)

model = nn.Sequential(fc1, relu, fc2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# linear model

linear = nn.Linear(input_size, num_classes)

criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(linear.parameters(), lr = lr)

l1 = []
l2 = []

# Training

for epoch in range(epochs):

    for i, (images, labels) in enumerate(train):

        images = Variable(images.view(-1, input_size))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        optimizer2.zero_grad()
        output2 = linear(images)
        loss2 = criterion2(output2, labels)
        loss2.backward()
        optimizer2.step()

    l1.append(loss.data[0])
    l2.append(loss2.data[0])

    print('After epoch: {0}, loss: {1}'.format(epoch, loss.item()))

print('Trinaing complete')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(list(range(epochs)), l1, label='MLP')
plt.plot(list(range(epochs)), l2, label='SLP')
plt.legend()
plt.grid(True)
plt.show()

with torch.no_grad():

    total = 0
    correct = 0
    correct2 = 0

    for (images, labels) in test:

        images = Variable(images.view(-1, input_size))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        outputs2 = linear(images)
        _,preds = torch.max(outputs2, 1)
        correct2 += (preds == labels).sum().item()
        

    print('Accuracy MLP : {}'.format(100*correct/ total))
    print('Accuracy SLP : {}'.format(100*correct2/ total))





        
