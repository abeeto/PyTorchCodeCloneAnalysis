import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np
class Network(nn.Module):
    def __init__(self, input_channel=3, output_size=4, softsign=False):
        super(Network, self).__init__()
        self.softsign = softsign
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)) 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(5 * 8 * 64, 512)
        self.dense2 = nn.Linear(512, output_size)
    def forward(self, input):
        output = F.relu(self.conv1(input.transpose(3, 2).transpose(2, 1)))
        output = F.relu(self.conv2(output))
        output = self.pool1(output)
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = self.pool2(output)
        output = F.relu(self.dense1(output.view(-1, 5 * 8 * 64)))
        output = self.dense2(output)
        if self.softsign:
            return F.softsign(output)
        else:
            return F.sigmoid(output)

