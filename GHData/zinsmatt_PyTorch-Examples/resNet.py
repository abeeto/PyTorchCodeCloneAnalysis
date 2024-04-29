#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:35:04 2021

@author: mzins
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt



class skip_connection(nn.Module):
    def __init__(self, channels, kernel):
        super(skip_connection, self).__init__()
        pad = (kernel - 1) // 2
        F1, F2, F3 = channels
        self.conv1 = nn.Conv2d(F1, F2, kernel, padding=pad)
        self.bn1 = nn.BatchNorm2d(F2)
        self.conv2 = nn.Conv2d(F2, F3, kernel, padding=pad)
        self.bn2 = nn.BatchNorm2d(F3)

    def forward(self, x):
        x_skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_skip
        return F.relu(x)


class skip_connection_div(nn.Module):
    """
        The first conv is 1x1 and the second kernel x kernel
    """
    def __init__(self, channels, kernel):
        super(skip_connection_div, self).__init__()
        F1, F2, F3 = channels
        self.conv1 = nn.Conv2d(F1, F2, 1, stride=2)
        self.bn1 = nn.BatchNorm2d(F2)
        pad = (kernel - 1) // 2
        self.conv2 = nn.Conv2d(F2, F3, kernel, padding=pad)
        self.bn2 = nn.BatchNorm2d(F3)

        self.conv_skip = nn.Conv2d(F1, F3, 1, stride=2)
        self.bn_skip = nn.BatchNorm2d(F3)

    def forward(self, x):
        x_skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x_skip = self.conv_skip(x_skip)
        x_skip = self.bn_skip(x_skip)
        x = x + x_skip
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(3, 64, 5, padding=2)
        self.bn_init = nn.BatchNorm2d(64)
        self.pool_init = nn.MaxPool2d(2, stride=2)

        self.backbone = nn.Sequential(*[
            skip_connection_div([64, 64, 128], 3),
            skip_connection([128, 64, 128], 3),
            skip_connection([128, 64, 128], 3),

            skip_connection_div([128, 128, 256], 3),
            skip_connection([256, 128, 256], 3),
            skip_connection([256, 128, 256], 3),
            skip_connection([256, 128, 256], 3),

            skip_connection_div([256, 256, 512], 3),
            skip_connection([512, 256, 512], 3),
            skip_connection([512, 256, 512], 3),
            skip_connection([512, 256, 512], 3),
            skip_connection([512, 256, 512], 3),
            skip_connection([512, 256, 512], 3)])

        self.avg = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.pool_init(x)
        x = self.backbone(x)
        x = self.avg(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


PATH = "resnet_weight.pth"
device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        transform=transform,
        download=True)
test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=32,
                                          shuffle=False)
n_epochs = 20
lr = 0.001

model = ResNet().to(device)
print(count_parameters(model), "parameters")
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(PATH))
for e in range(n_epochs):
    for X, Y in train_loader:
        X_dev = X.to(device)
        Y_dev = Y.to(device)

        optimizer.zero_grad()
        pred = model(X_dev)
        loss = loss_fn(pred, Y_dev)

        loss.backward()
        optimizer.step()

    print(loss.item())

torch.save(model.state_dict(), PATH)

#%%
with torch.no_grad():
    model.eval()
    nb_good = 0
    nb_total = 0
    for X, Y in test_loader:
        X_dev = X.to(device)
        pred = model(X_dev)
        pred_classes = torch.argmax(pred.cpu(), dim=1)
        nb_total += pred_classes.shape[0]
        nb_good += torch.sum(pred_classes == Y).item()

    print("Accuracy = %.3f (%d / %d)" % (100*nb_good/nb_total, nb_good, nb_total))