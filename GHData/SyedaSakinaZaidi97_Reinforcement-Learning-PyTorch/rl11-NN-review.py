#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

UPDATE 24/May/2018
code was not CPU/GPU ready (only worked on CPU)
so adjusted to be really generic

@author: udemy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

W = 2
b = 0.3

x = Variable(torch.arange(100).unsqueeze(1))
if use_cuda:
    x = x.cuda()

y = W * x + b

###### PARAMS ######
learning_rate = 0.01
num_episodes = 1000


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(1,1)
        
    def forward(self, x):
        output = self.linear1(x)
        return output
    
mynn = NeuralNetwork()

if use_cuda:
    mynn.cuda()
    
loss_func = nn.MSELoss()
#loss_func = nn.SmoothL1Loss()

optimizer = optim.Adam(params=mynn.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

for i_episode in range(num_episodes):
    
    predicted_value = mynn(x)
    
    loss = loss_func(predicted_value, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i_episode % 50 == 0:
        print("Episode %i, loss %.4f " % (i_episode, loss.data[0]))
    
    
plt.figure(figsize=(12,5))
plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), alpha=0.6, color='green')
plt.plot(x.data.cpu().numpy(), predicted_value.data.cpu().numpy(), alpha=0.6, color='red')

plt.show()

