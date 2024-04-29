# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:45:09 2020

@author: ashwin
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dataset for MNIST numbers
class MNIST_Data(Dataset):
    def __init__(self, imgfile, labelfile):
        print("--- Reading Images ---")
        f = open (imgfile, "rb")
        magicNum = f.read(4)
        self.size = int.from_bytes(f.read(4), "big")
        self.imgrows = int.from_bytes(f.read(4), "big")
        self.imgcols = int.from_bytes(f.read(4), "big")
        self.data = []
        
        #read the image data
        b = f.read(1)
        while(b != b""):
            self.data.append(b)
            b = f.read(1)
            #print(b)
        self.data = torch.ByteTensor(self.data)
        self.data = self.data.view(-1, self.imgcols*self.imgrows)
        self.data = self.data.float()/255
        
        #read the label data
        print("--- Reading Labels ---")
        f2 = open(labelfile, "rb")
        magicNum = f2.read(4)
        print(int.from_bytes(f2.read(4), "big")) #throw away duplicate size
        b = f2.read(1)
        self.labels = []
        while(b != b""):
            self.labels.append(b)
            b = f2.read(1)
            
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], int.from_bytes(self.labels[idx], "big")
    def viewItem(self, idx, loud=True):
        lab = int.from_bytes(self.labels[idx], "big")
        if(loud):
            print("Viewing img: " + str(idx) + " with label: " + str(lab))
        plt.imshow(self.data[idx].view(28, 28).numpy(), cmap='gray')
        plt.show()  
        

print("[TRAIN]")
train = MNIST_Data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
print("[TEST]")
test = MNIST_Data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

# %%
class MNIST_Network(torch.nn.Module):
    def __init__(self, insize, hiddensize, outsize):
        super(MNIST_Network, self).__init__()
        self.linear1 = torch.nn.Linear(insize, hiddensize)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hiddensize, outsize)
    def forward(self, x):
        hidden = self.linear1(x)
        hiddenAct = self.relu(hidden)
        out = self.linear2(hiddenAct)
        return out
# %%

#sanity check on data loaders
test.viewItem(0)
test.viewItem(9)
train.viewItem(5)

# setup
trainloader = DataLoader(train, batch_size=32)
net = MNIST_Network(28*28, 32, 10)
lossFunc = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters())

# %%
# training 
for epoch in range(0, 3):
    i = 0
    for x_batch, y_batch in trainloader:
        #print(y_batch)
        y_pred = net(x_batch)
        loss = lossFunc(y_pred, y_batch)
        loss.backward()
        
        optim.step()
        optim.zero_grad()
        if(i == 0): 
            print(loss)
        i+=1
        #with torch.no_grad():
        #    for param in net.params()
        #        param -= lr*param.grad
# %%
# eval
testloader = DataLoader(test, batch_size=1)
cor = 0
total = 0
with torch.no_grad():
    for x, y in testloader:
        y_pred = net(x)
        y_pred_max = torch.max(y_pred, 1).indices
        #print(str(y_pred_max) + " --> " + str(y))
        if(y == y_pred_max):
            cor+=1
        total+=1
    print(cor/total)
    
    import random
    randomSamples = [random.randrange(0, 10000) for i in range(10)]
    sample = 2343
    for sample in randomSamples:
        x, y = test[sample]
        y_pred = net(x)
        y_pred_max = torch.max(y_pred, 0).indices
        print("pred: " + str(y_pred_max) + " actual: " + str(y))
        test.viewItem(sample, False)