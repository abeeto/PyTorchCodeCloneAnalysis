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
        self.lin = nn.Linear(28 * 28, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.lin(x)
        x = self.logSoftmax(x)
        return x # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.inputLayer = nn.Linear(28 * 28, 10 * 15)
        self.hiddenLayer = nn.Linear(10 * 15, 10 * 7)
        self.outputLayer = nn.Linear(10 * 7, 10)
        self.hyperbolicTan = nn.Tanh()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.inputLayer(x)
        x = self.hyperbolicTan(x)
        x = self.hiddenLayer(x)
        x = self.hyperbolicTan(x)
        x = self.outputLayer(x)
        x = self.logSoftmax(x)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.convolutionalLayer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.convolutionalLayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.hiddenLayer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 10 * 50),
            nn.ReLU()
        )
        self.outputLayer = nn.Linear(10 * 50, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convolutionalLayer1(x)
        x = self.convolutionalLayer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.hiddenLayer(x)
        x = self.outputLayer(x)
        x = self.logSoftmax(x)
        return x # CHANGE CODE HERE
