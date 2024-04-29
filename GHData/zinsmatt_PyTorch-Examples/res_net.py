#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:14:26 2019

@author: matt
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 80
lr = 0.001

transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

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
                                           batch_size=100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=100,
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, n_blocks, n_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(block, 16, n_blocks[0])
        self.layer2 = self.make_layer(block, 32, n_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, n_blocks[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, n_classes)
    
    def make_layer(self, block, out_channels, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride),
                    nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride,
                            downsample=downsample))
        self.in_channels = out_channels
        for i in range(1, n_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer3(self.layer2(self.layer1(out)))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
        
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train
cur_lr = lr
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
    if (e+1) % 20 == 0:
        cur_lr /= 3
        update_lr(optimizer, cur_lr)
    
# test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (pred == labels).sum()
    print("Test accuracy = %.3f" % (100 * correct / total))
