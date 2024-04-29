#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:02:57 2019

@author: matt
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

n_epochs = 3
n_classes = 10
batch_size = 100
lr = 0.001


train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True)

test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        temp_layers = []        
        temp_layers.append(nn.Conv2d(1, 16,
                                     kernel_size=5,
                                     stride=1,
                                     padding=5))
        
        temp_layers.append(nn.BatchNorm2d(16))
        temp_layers.append(nn.ReLU())
        
#        temp_layers.append(nn.Conv2d(8, 16,
#                                     kernel_size=5,
#                                     stride=1,
#                                     padding=2))
#        temp_layers.append(nn.BatchNorm2d(16))
#        temp_layers.append(nn.ReLU())
        
        temp_layers.append(nn.MaxPool2d(kernel_size=2,
                                        stride=2))
        temp_layers.append(nn.Conv2d(16, 32,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2))
        temp_layers.append(nn.BatchNorm2d(32))
        temp_layers.append(nn.ReLU())
        temp_layers.append(nn.MaxPool2d(kernel_size=2,
                                        stride=2))
        temp_layers.append(nn.Linear(8*8*32, 8*8*16))        
        temp_layers.append(nn.ReLU())
        temp_layers.append(nn.Linear(8*8*16, 8*8*8))        
        temp_layers.append(nn.ReLU())
        temp_layers.append(nn.Linear(8*8*8, 8*8*2))        
        temp_layers.append(nn.ReLU())
        temp_layers.append(nn.Linear(8*8*2, 1))        
        temp_layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(temp_layers)
        
    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            out = self.layers[i](out)
            if i == 7:
                out = out.reshape(out.size(0), -1)
#        out = self.layer2(self.layer1(x))
#        print(out.shape)
        
#        print(out.shape)
#        out = self.layers[-2](out)
#        print(out.shape)
        return out #{self.layers[-1](out)

model = ConvNet(n_classes).to(device)

model_state = model.state_dict()

pretrained_weights = torch.load("saved_state.pth")
pretrained_weights = {k : v for k, v in pretrained_weights.items() if k in model_state.keys() and k != "layers.14.bias" and k != "layers.14.weight" }
print(pretrained_weights.keys())
print(model_state.keys())
model_state.update(pretrained_weights)
model.load_state_dict(model_state)


#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

save = []
# Train
for e in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()   

        outputs = model(images)
        loss = loss_fn(outputs*10, labels.float())
        
        save.append(model.layers[0].weight.cpu().detach().numpy().copy())
        
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch %d batch %d loss %f" % (e, i, loss.item()))
    if e == 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 3
    elif e == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 3
kernels2 = [t.squeeze() for t in save]


for i in np.linspace(0, len(kernels2)-1, 40, dtype=int):
    plt.figure(i)
    plt.imshow(kernels2[i][0, :, :])
    plt.savefig("output/kernel_%04d.png" % i)
    plt.close()
    
K = kernels2[0].shape[0]
for i in range(K):
    plt.figure(i)
    plt.imshow(kernels2[0][i, :, :])
    plt.savefig("output/init_%03d.png" % i)
    plt.close()
    plt.figure(i)
    plt.imshow(kernels2[-1][i, :, :])
    plt.savefig("output/end_%03d.png" % i)
    plt.close()
    

print("Total of %d weights" % sum([p.numel() for p in model.parameters()]))
    
#%%

#print("Model state_dict:")
#for param in model.state_dict():
#    print(param, "\t", model.state_dict()[param].size())
#    
#for p in model.parameters():
#    print(p.size())
#    
#print("Optimizer parameters:")
#for param in optimizer.state_dict():
#    print(param, "\t", optimizer.state_dict()[param])
    
#torch.save(model.state_dict(), "saved_state.pth")

#%%


# Test
model.eval()    # batch norm uses mean/variance instead
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # get predicted classes
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print("Test accuracy = %.2f" %(100 * correct / total))

#torch.save(model.state_dict(), "model.ckpt")
