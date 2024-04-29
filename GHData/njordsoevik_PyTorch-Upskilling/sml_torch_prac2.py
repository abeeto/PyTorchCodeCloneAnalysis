# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:35:50 2020

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
target_y=torch.Tensor([0,1,1,0])
inputs = x
labels = target_y
train = TensorDataset(inputs,labels) # here we're just putting our data samples into a tiny Tensor dataset

trainloader = DataLoader(train, batch_size=2, shuffle=False) # and then putting the dataset above into a data loader
# the batchsize=2 option just means that, later, when we iterate over it, we want to run our model on 2 samples at a time

linear_layer1=nn.Linear(4,1)

epochs=4
lr=1e-4
loss_function=nn.MSELoss()
optimizer=optim.SGD(linear_layer1.parameters(),lr=lr)

for e in range(epochs):
    train_loader_iter=iter(trainloader) 
    for batch_idx, (inputs,labels) in enumerate(train_loader_iter): # here we split apart our data so we can run it
        linear_layer1.zero_grad()
        inputs,labels=Variable(inputs.float()), Variable(labels.float())
        predicted_y=linear_layer1(inputs)
        loss = loss_function(predicted_y,labels)
        loss.backward()
        optimizer.step()
        print("----------------------------------------")
        print("Output (UPDATE: Epoch #" + str(e + 1) + ", Batch #" + str(batch_idx + 1) + "):")
        print(linear_layer1(Variable(x)))
        print("Should be getting closer to [0, 1, 1, 0]...") # but some of them aren't! we need a model that fits better...
                                                             # next up, we'll convert this model from regression to a NN

print("----------------------------------------")
