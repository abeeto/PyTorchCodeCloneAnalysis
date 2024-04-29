# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:52:54 2020

@author: DELL
"""
=============================================================================================================================
#  First successful implementation of PyTorch functions for logistic regression, based on the PyTorch tutorial.
=============================================================================================================================

#%%

# Logistic Regression : Tutorial code

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

start = time.time()

# Hyper-parameters 
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

print(train_loader)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)
        # Forward pass
        outputs = model(images)
        #print(outputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

print()
print()  
end = time.time()
  
print('Total time required : ', end - start, 's')

#%%

# Logistic Regression : Code I developed based on tutorial code.
    
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time()

input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.01
 
   
def inputvector(n):    # extracting a batch of 100 images
    data = pd.io.parsers.read_csv("mnist_train.csv", skiprows=range(0,n), nrows=100)
    data = np.matrix(data, dtype = np.float32)
    n += 100
    return data,n


def inputforvalidation():  # extracting a batch of 10000 images for  validation
    data = pd.io.parsers.read_csv("mnist_train.csv", skiprows=range(0,35000), nrows=10000)
    data = np.matrix(data, dtype = np.float32)
    return data


def dataprocessing(xl): #Seperating labels and image data from the data matrix
    s = xl.shape[1]
    x1 = [[0 for x in range(s-1)] for y in range(len(xl))] 
    x1 = np.matrix(x1, dtype = np.float32)
    y1 = [[0] for y in range(len(xl))]
    y1 = np.matrix(y1)
    for i in range(len(xl)):
        x1[i,:] = xl[i, 1:]
        y1[i] = xl[i,0]
    return x1/1000, y1


# Logistic regression model
model = nn.Linear(input_size, num_classes)


# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


n = 0   # number of images taken from 


print()
print('Training.....')
print()


for epoch in range(num_epochs):
    images, n = inputvector(n)     # Extracting image vectors from MNIST dataset for training
    images, labels = dataprocessing(images)     # Seperating images and labels
    images_tensor = torch.from_numpy(images)    # Creating tensor of images
    labels_tensor = torch.from_numpy(labels)    # Creating tensor of labels
    labels_tensor = labels_tensor.long()    # Coverting labels to long from int
    for counter in range(150):
        for i in range(len(images)):
            output = model(images_tensor[i].reshape(-1, input_size))    # Reshape images to (batch_size, input_size) and forward pass
            cost = loss_function(output, labels_tensor[i])     # Calculating cost
            # Backward and optimize
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        if (counter+1)%50 == 0:
            print('Epoch =', epoch + 1, '/ 5,  Step :', counter + 1, '/ 150,  Cost =', cost.item())
    print()
    
    
print()
print('Evaluating....') 

images_test = inputforvalidation()      # Extracting 10000 image vectors from MNIST dataset for validation
images_test, labels_test = dataprocessing(images_test)  # Seperating images and labels
images_test_tensor = torch.from_numpy(images_test)  # Creating tensor of images
labels_test_tensor = torch.from_numpy(labels_test)  # Creating tensor of labels
labels_test_tensor = labels_test_tensor.long()  # Coverting labels to long from int
accuracy = 0
y = 0

with torch.no_grad():
    for i in images_test_tensor:
        output = model(i.reshape(-1, input_size))
        _, predicted = torch.max(output.data, 1)
        accuracy += (predicted == labels_test_tensor[y])
        y += 1
print('Accuracy = ', accuracy.item()*100/len(images_test), '%')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

print()
print()  
end = time.time()

# Time required for execution  
print('Total time required : ', round((end - start), 2), 's')
