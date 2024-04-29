#!/usr/bin/env python

import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


image, label = next(iter(trainloader))
helper.imshow(image[0,:]);


# Defining network architecture 
from torch import nn

model = nn.Sequential(nn.Linear(784,342),
                     nn.ReLU(),
                     nn.Linear(342,172),
                     nn.ReLU(),
                     nn.Linear(172,64),
                     nn.ReLU(),
                     nn.Linear(64,10),
                     nn.LogSoftmax(dim=1)
                     )


#Creating the network, defining the criterion and optimizer
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=.003)


#Training the network 
epochs = 5
for e in range(epochs):
    running_loss = 0
    
    for images,labels in trainloader:
        images = images.view(images.shape[0],-1)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        
        loss = criterion(output,labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss+=loss.item()
    else: 
        print('Training loss',running_loss)



get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import helper
import torch.nn.functional as F
# Testing the network

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

logits = model.forward(img)
# Calculate the class probabilities (softmax) for img
ps = F.softmax(logits,dim=1)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')

