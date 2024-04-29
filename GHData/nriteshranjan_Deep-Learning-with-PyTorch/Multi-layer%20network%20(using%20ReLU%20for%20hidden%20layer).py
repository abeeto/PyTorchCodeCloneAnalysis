# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:49:55 2020

@author: rrite
"""

from torch import nn
import torch

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
                
    def forward(self,x):
        # Hidden layer with sigmoid activation
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim = 1)

model = Network()
model