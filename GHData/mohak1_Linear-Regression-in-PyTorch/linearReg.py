#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:58:55 2018

Linear Regression in PyTorch

@author: mohak
"""

#Step1: import the libraries and data
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

xTrain = Variable(torch.Tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]).cuda())
yTrain = Variable(torch.Tensor([[10], [20], [30], [40], [50], [60] ,[70], [80], [90], [100]]).cuda())

#Step2: Create a class
class LinearReg(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearReg,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
        

#Step3: Instanciate the class
input_dim=1
output_dim=1
model = LinearReg(input_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()

#Step4: Instanciate loss class
loss = nn.MSELoss()

#Step5: Instantiate the optimizer class
learning_rate=0.01
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Step6: Epochs for loop
epochs=1000
for epoch in range(epochs):
    epoch = epoch+1
    inp = xTrain
    out = yTrain
    yHat = model(xTrain)
    er = loss(yHat,yTrain)
    optim.zero_grad()
    er.backward()
    optim.step()
    if(epoch%10==0):
        print('epochs = {0}, loss={1}'.format(epoch, er))
   
print(model(yTrain))

#Not saving this model
