# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:07:56 2020

@author: NjordSoevik
"""

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
trainloader = DataLoader(train, batch_size=4, shuffle=False) # and then putting the dataset above into a data loader
# the batchsize=2 option just means that, later, when we iterate over it, we want to run our model on 2 samples at a time
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2) # here's where we define the same layers we had earlier
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x) # the forward function just sends everything through its respective layers
        x = self.sigmoid(x) # including through the sigmoids after each Linear layer
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

net = Net() # we made a blueprint above for our neural network, now we initialize one.

epochs = 20
LEARNING_RATE = 1e-2
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE) # slight difference: we optimize w.r.t. the net parameters now


for epoch in range(epochs):
    train_loader_iter = iter(trainloader)
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        net.zero_grad() # same here: we have to zero out the gradient for the neural network's inputs
        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        output = net(inputs) # but now, all we have to do is pass our inputs to the neural net 
        
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        print("----------------------------------------")
        print("Output (UPDATE: Epoch #" + str(epoch + 1) + ", Batch #" + str(batch_idx + 1) + "):")
        print(net(Variable(x))) # much better!
        print("Should be getting closer to [0, 1, 1, 0]...")

print("----------------------------------------")