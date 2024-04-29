# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:55:01 2020

@author: NjordSoevik
"""
import torch # Tensor Package (for use on GPU)
import torch.optim as optim # Optimization package
import matplotlib.pyplot as plt # for plotting
import numpy as np
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
from torch.autograd import Variable # for computational graphs

x = torch.Tensor([[0, 0, 1, 1],
                 [0, 1, 1, 0],
                 [1, 0, 1, 0],
                 [1, 1, 1, 1]])
target_y=torch.Tensor([0,0,1,1])
inputs = x
labels = target_y
train = TensorDataset(inputs,labels) # here we're just putting our data samples into a tiny Tensor dataset

trainloader = DataLoader(train, batch_size=2, shuffle=False) # and then putting the dataset above into a data loader
# the batchsize=2 option just means that, later, when we iterate over it, we want to run our model on 2 samples at a time

linear_layer1=nn.Linear(4,6)
sigmoid=nn.Sigmoid()
linear_layer2=nn.Linear(6,1)

epochs=4
lr=1e-3
loss_function=nn.MSELoss()
optimizer=optim.SGD(linear_layer1.parameters(),lr=lr)

for epoch in range(epochs):
    train_loader_iter=iter(trainloader)
    for batch_idx, (inputs,labels) in enumerate(train_loader_iter):
        linear_layer1.zero_grad()
        inputs,labels=Variable(inputs.float()), Variable(labels.float())
        
        linear_layer1_output=linear_layer1(inputs)
        sigmoid_output1=sigmoid(linear_layer1_output)
        linear_layer2_output=linear_layer2(sigmoid_output1)
        sigmoid_output2=sigmoid(linear_layer2_output)
        
        loss=loss_function(sigmoid_output2,labels)
        loss.backward()
        optimizer.step()
        
        print("----------------------------------------")
        print("Output (UPDATE: Epoch #" + str(epoch + 1) + ", Batch #" + str(batch_idx + 1) + "):")
        print(sigmoid(linear_layer2(sigmoid(linear_layer1(Variable(x)))))) # the nested functions are getting out of hand..
        print("Should be getting closer to [0, 1, 1, 0]...") # they are if you increase the epochs amount... but it's slow!