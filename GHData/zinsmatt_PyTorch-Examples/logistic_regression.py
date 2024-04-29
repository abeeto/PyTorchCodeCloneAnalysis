#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:22:00 2019

@author: matt
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


size_in = 784
n_classes = 10
n_epochs = 5
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

model = nn.Linear(size_in, n_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train
total_step = len(train_loader)
for e in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape((-1, 28 * 28))

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch %d batch %d loss %f" % (e, i, loss.item()))

# Test
with torch.no_grad():
    ok = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        ok += (pred == labels).sum()
    print("Test accuracy = %.1f" % (100 * ok / total))