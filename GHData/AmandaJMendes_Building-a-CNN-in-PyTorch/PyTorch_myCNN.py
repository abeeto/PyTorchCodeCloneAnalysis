#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:53:32 2022

@author: amanda
"""

import torch
import torchvision

"""
DATA PREPARATION
"""
bs = 256
train_split = 0.8

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((16, 16)),
                                            torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

n = len(train_set)
n_train = int(train_split*n)
n_val = int(n - n_train)

train_set, validation_set = torch.utils.data.random_split(dataset = train_set,
                                                          lengths = [n_train, n_val])


train = torch.utils.data.DataLoader(dataset = train_set, batch_size = bs)
validation = torch.utils.data.DataLoader(dataset = validation_set, batch_size = bs)
test = torch.utils.data.DataLoader(dataset = test_set, batch_size = bs)


"""
MODEL
"""
class MyCNN(torch.nn.Module):
    def __init__(self, *args):
        super(MyCNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        if not args:
            args = [1, 32, 10]
            
        self.n_fc = 16-(len(args)-2) 
        n_in = args[0]
        n_out = args[-1]
        
        for i in args[1:-1]:
            self.layers.append(torch.nn.Conv2d(n_in, i,
                                              kernel_size = 5,
                                              padding = 'same'))
            self.layers.append(torch.nn.BatchNorm2d(i))
            self.layers.append(torch.nn.ReLU(i))
            self.layers.append(torch.nn.MaxPool2d(kernel_size = 2, stride = 1))
            n_in = i
        self.layers.append(torch.nn.Linear(args[-2]*self.n_fc**2, n_out))
        self.layers.append(torch.nn.BatchNorm1d(n_out))
        
    def forward(self, img_tensor):
        out = img_tensor
        for l in self.layers[:-2]:
            out = l(out)
        out = out.view((out.size(0), -1))
        out = self.layers[-2](out)
        out = self.layers[-1](out)
        return out
    
    
"""
TRAINING
"""
model = MyCNN(1, 16, 32, 10)
loss_func = torch.nn.CrossEntropyLoss()
lr = 0.001
opt = torch.optim.Adam(model.parameters(), lr = lr)
epochs = 20

costs = []
accs = [0]

for i in range(epochs):
    
    cost = 0
    model.train()
    for img, label in train:
        out = model(img)
        loss = loss_func(out, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        cost += loss.data
    costs.append(cost)
    
    correct=0
    model.eval()
    for img, label in validation:
        out = model(img)
        _, out_idx = torch.max(out, 1)
        correct += (label == out_idx).sum().item()
    accs.append(correct / len(validation_set))
    print(accs[-1])

    if accs[-1]<accs[-2]:
        break
    
    
"""
TESTING
"""
model.eval()    
correct = 0             
for img, label in test:
    out = model(img)
    _, out_idx = torch.max(out, 1)
    correct += (label == out_idx).sum().item()
print("Accuracy in the testing set: ", correct / len(test_set))
        
        
