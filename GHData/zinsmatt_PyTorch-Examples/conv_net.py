#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:02:57 2019

@author: matt
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

n_epochs = 5
n_classes = 10
batch_size = 100
lr = 0.001


train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True)

test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2))
        self.fc = nn.Linear(7*7*32, n_classes)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

model = ConvNet(n_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
for e in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch %d batch %d loss %f" % (e, i, loss.item()))

# Test
model.eval()    # batch norm uses mean/variance instead
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # get predicted classes
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print("Test accuracy = %.2f" %(100 * correct / total))

torch.save(model.state_dict(), "model.ckpt")
