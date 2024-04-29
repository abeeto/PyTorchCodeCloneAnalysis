import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class FCN(nn.Module):
    def __init__(self, h, w, out):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.size1 = [(h-4)//2, (w-4)//2]

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.size2 = [(self.size1[0]-4)//2, (self.size1[1]-4)//2]

        self.conv3 = nn.Conv2d(32, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.size3 = [(self.size2[0]-4)//2, (self.size2[1]-4)//2]

        self.conv4 = nn.Conv2d(64, 128, self.size3)
        # self.dense1 = nn.Linear(self.size3[0]*self.size3[1]*64, 128)
        self.dense2 = nn.Linear(128, out)

        self.layers = [
            self.conv1, self.bn1, self.pool1, func.relu,
            self.conv2, self.bn2, self.pool2, func.relu,
            self.conv3, self.bn3, self.pool3, func.relu,
            self.conv4, func.relu,
            nn.Flatten(),
            self.dense2
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResNet(nn.Module):
    def __init__(self, h, w, out):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.BatchNorm2d(16)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.size1 = [h//2, w//2]

        self.conv2 = nn.Conv2d(16, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.size2 = [self.size1[0]//2, self.size1[1]//2]

        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128)
        )
        self.size3 = self.size2

        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, out)
    
    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = self.pool1(func.relu(self.shortcut1(x) + out1))
        out2 = self.bn2(self.conv2(out1))
        out2 = self.pool2(func.relu(self.shortcut2(out1) + out2))
        out3 = self.bn3(self.conv3(out2))
        out3 = func.relu(self.shortcut3(out2) + out3).view(-1, 128, self.size3[0]*self.size3[1])
        out3 = torch.max(out3, 2)[0]
        out = func.relu(self.dense1(out3))
        out = self.dense2(out)
        return out

class FullyConnected(nn.Module):
    def __init__(self, input, out):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
