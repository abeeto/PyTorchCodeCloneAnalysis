# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:13:45 2020

@author: rrite
"""
from IPython import get_ipython
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import helper
import torch.nn.functional as F
import matplotlib.pyplot as plt
import fc_model

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
image, label = next(iter(trainloader))
helper.imshow(image[0, :])

# fc_model : for creating a fully-connected network
# Create the network, define the criterion and optimiser

model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)
fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs = 2)

# Saving and Loading network
print("Our model: \n\n", model, '\n')
print(("The state dict keys: \n\n", model.state_dict().keys()))
checkpoint = {'input_size'    : 784,
              'output_size'   : 10,
              'hidden_layers' : [each.out_features for each in model.hidden_layers],
              'state_dict'    : model.state_dict()}
torch.save(model.state_dict(), 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint('checkpoint.pth')
print(model)