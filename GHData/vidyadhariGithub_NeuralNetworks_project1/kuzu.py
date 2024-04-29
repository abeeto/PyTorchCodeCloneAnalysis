# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.input = torch.nn.Linear(784,10)

    def forward(self, x):
        #for epoch in range(1, 10 + 1):
        x = x.reshape(x.size(0), -1)
        out = self.input(x)
        out = F.log_softmax(out, dim=1)
        return out
        #return 0 # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.input = torch.nn.Linear(784, 210)
        self.output = torch.nn.Linear(210, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.input(x)
        x = torch.tanh(x)
        x = self.output(x)
        out = F.log_softmax(x, dim=1)
        return out
        #return 0 # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        #first convolutional layer with 1 in_channel, 16 out_channels, kernel size 5
        self.conv1 = torch.nn.Conv2d(1, 16, 5, 1, 2)
        #second convolutional layer with 16 in_channels, 32 out_channels, kernel size 5
        self.conv2 = torch.nn.Conv2d(16, 32, 5, 1, 2)
        self.pool = torch.nn.MaxPool2d(5, padding = 2) #sub-sampling
        #fully connected layer
        self.fc1 = torch.nn.Linear(1152, 650)
        #output layer, 64 inputs and 10 classes from 0-9
        self.fc2 = torch.nn.Linear(650, 10)

    def forward(self, x):
        x = self.conv1(x) #convolution layer
        x = F.relu(x)
        x = self.conv2(x) #convolution layer
        x = F.relu(x)
        x = self.pool(x) #max pooling
        y = x.view(x.shape[0], -1)
        y = self.fc1(y) #fully connected layer
        y = F.relu(y)
        y = self.fc2(y) #output layer
        out = F.log_softmax(y, dim=1)
        return out
        #return 0 # CHANGE CODE HERE
