#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

device = torch.device("cuda:0" if use_cuda else "cpu")

W = 2
b = 0.3

#x = torch.arange(100,dtype=torch.float32).to(device).unsqueeze(1)
x = torch.arange(100).to(device).unsqueeze(1)

print(x.type()) 

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
    
mynn = NeuralNetwork().to(device)
    
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
        print("Episode %i, loss %.4f " % (i_episode, loss.item()))
    
    
plt.figure(figsize=(12,5))
plt.plot(x.cpu().numpy(), y.cpu().numpy(), alpha=0.6, color='green')
plt.plot(x.cpu().numpy(), predicted_value.detach().cpu().numpy(), alpha=0.6, color='red')

if use_cuda:
    plt.savefig("graph.png")
else:
    plt.show()

