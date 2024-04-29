#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:54:09 2021

@author: jeonsohyun
"""
from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0],[2.0],[3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])
learning_rate = 0.01
epoch_num = 500


## Step 1. Design Model
    
class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__()           # initiate parent
        self.linear = torch.nn.Linear(1,1)      # instantiate linear model
        
    def forward(self, x):
        '''
        Accept variable as input data, and return variable as output data.
        '''
        y_pred = self.linear(x)
        return y_pred


def training(model,x_data, y_data, training_num, criteron,optim_fun):
    for epoch in range(training_num):
        # 1. Forward pass : compute predicted y by passing x to the model
        y_pred = model(x_data)
        
        # 2. Compute and print loss
        loss = criterion(y_pred, y_data)
        if epoch%100 == 0:
            print(f'Epoch: {epoch} | Loss: {loss.item()}')
        
        # zero gradient, perform a backward pass, and update the weight
        optim_fun.zero_grad()
        loss.backward()
        optim_fun.step()
        
    return model


# Make model instance
model = Model()


## Step2. Construct loss and optimizer
# model.parameters() in the SGD constructor will contain the learnable parameters
# of the tow nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


## Step3. Training Loop
for epoch in range(epoch_num):
    # 1. Forward pass : compute predicted y by passing x to the model
    y_pred = model(x_data)
    
    # 2. Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()}')
    
    # zero gradient, perform a backward pass, and update the weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = tensor([[4.0]]) # new x value
y_pred = model(hour_var)
print(f'Prection (after training) : {hour_var.item()}, {y_pred.item()}')



### Excercise: Apply various of optimization function.
   
# Make model instance
new_val = tensor([[5.0]])

# Adam
model = Model()
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = training(model, x_data, y_data, epoch_num, criterion, optimizer_Adam)
print(f'Adam Predict with new value : {new_val}, {model(new_val).item()}')

# Adagrad
model = Model()
optimizer_Adagrad = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
model = training(model, x_data, y_data, epoch_num, criterion, optimizer_Adagrad)
print(f'Adagrad Predict with new value : {new_val}, {model(new_val).item()}')

# RMSprop
model = Model()
optimizer_RMSprop = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
model = training(model, x_data, y_data, epoch_num, criterion, optimizer_RMSprop)
print(f'RMSprop Predict with new value : {new_val}, {model(new_val).item()}')

#Rprop
model = Model()
optimizer_Rprop = torch.optim.Rprop(model.parameters(), lr=learning_rate)
model = training(model, x_data, y_data, epoch_num, criterion, optimizer_Rprop)
print(f'Rporp Predict with new value : {new_val}, {model(new_val).item()}')



