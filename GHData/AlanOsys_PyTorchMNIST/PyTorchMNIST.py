#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

train = datasets.MNIST("",train = True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#dataset for training


test = datasets.MNIST("",train = False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#dataset for testing the viability of the nn

#setup for sets:
#setting the batches the sets will be executed in and the fact they will be shuffled to reduce sequence learning


# In[ ]:


import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

train = datasets.MNIST("",train = True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#dataset for training


test = datasets.MNIST("",train = False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#dataset for testing the viability of the nn

#setup for sets:
#setting the batches the sets will be executed in and the fact they will be shuffled to reduce sequence learning

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle=True)
print(trainset)
class Net(nn.Module):
    arr = np.zeros(200)
    def __init__(self):#Sets up the neurons and their sizes
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)#input of 28*28 which is the size of the picture
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)#output is only needed to be 10 because we have numbers from 0-9
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x,dim=1)
        
net = Net()

optimizer = optim.Adam(net.parameters(),lr=0.001)#optimizer function

EPOCHS = 3

for epock in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()#resets the gradient
        output = net(X.view(-1, 28*28))#calculates the current output by processing 'data' through the network 
        loss = F.nll_loss(output, y)#calculates loss
        loss.backward()#backpropagates to get back lost weight
        optimizer.step()


plt.imshow(X[1].view(28,28))
plt.show()
print(torch.argmax(net(X[1].view(-1,784))))

