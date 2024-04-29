#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:40:05 2019

@author: matt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print("after conv1: ", x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        print("after conv2: ", x.size())
        x = x.view(-1, self.num_flat_features(x))
        print("after view: ", x.size())
        x = F.relu(self.fc1(x))
        print("after fc1: ", x.size())
        x = F.relu(self.fc2(x))
        print("after fc2: ", x.size())
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        print("num_flat_features returns ", num_features)
        return num_features

net = Net()
print(net)
params = list(net.parameters())
print(len(params))

input_data = torch.randn(1, 1, 32, 32)

output = net(input_data)

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
net.zero_grad()

print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.001)

optimizer.zero_grad()
out = net(input_data)
loss = criterion(out, target)
print(loss)
loss.backward()
optimizer.step()

out = net(input_data)
loss = criterion(out, target)
print(loss)
