#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:16:17 2021

@author: jeonsohyun
"""
from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim


# Data set
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])
learning_rate =0.01 
epoch_num = 1000


## Step1. Design Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)        # one in and one out
        
    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred
    
    
# model instance
model = Model()

## Step2. Construct criterion and optimization
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


## Step3. Training Cycle
for epoch in range(epoch_num):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    if epoch%100==0:
        print(f'Epoch : {epoch}/{epoch_num} | Loss :{loss.item()} ')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
# After training
new_val = tensor([[1.0]])
print(f'Predict : {new_val.item()}, {model(new_val).item()}')

new_val = tensor([[7.0]])
print(f'Predict : {new_val.item()}, {model(new_val).item()}')

