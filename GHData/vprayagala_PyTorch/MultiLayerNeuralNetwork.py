# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:27:31 2018

@author: vprayagala2

Stack the layers up and calculate the hidden units values

This is an untrained model to calculate score of output
"""
#%%
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
#%%
def sigmoid_activation(x):
    """
        Sigmoid Activation Function for a Neuron
        
        Input: x - Torch Tensor
    """
    return (1/(1+np.exp(-x)))

def softmax(x):
    """
        Softmax Activation Function for a Neuron
        e,x / sum(e,x)
        Input: x - Torch Tensor
    """   
    return torch.exp(x) / (torch.sum(torch.exp(x),dim=1).view(-1,1))
    
#%%
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),
                                                   (0.5,0.5,0.5))
                              ])
#Download the mnist data
train_data=datasets.MNIST("C:\Data\MNIST-Data\\",download=True,train=True,transform=transform)
train_loader=torch.utils.data.DataLoader(train_data,
                                         batch_size=32,
                                         shuffle=True)

dataiter=iter(train_loader)
image, labels = dataiter.next()
print("Image Shape:{}".format(image.shape))
print("Label Shape:{}".format(labels.shape))
#%%
#Generate Some Random Data
# Image Shape is 64,1,28,28 - batch of 64 images with one color chanel of 28*28 pixels

#We need to flatten the image to 1D tensor with 28*28 pixels and 64 such images
train_img=image.view(image.shape[0],-1)
n_input=train_img.shape[1]
n_hidden=256
n_output=10

#Weight for input to hidden layer
W1=torch.randn(n_input,n_hidden)
#Weights for hidden to output layer
W2=torch.randn(n_hidden,n_output)
print("Input Tensor Shape:%s Rows %s Columns"%(train_img.shape))
print("Input Weight Matrix:%s Rows %s Columns"%(W1.shape))
print("Hidden Weight Matrix:%s Rows %s Columns"%(W2.shape))

#Bias for hidden and ouput
b1=torch.randn(1,n_hidden)
b2=torch.randn(1,n_output)
print("Input Bias Shape:%s Rows %s Columns"%(b1.shape))
print("Hidden Bias Shape:%s Rows %s Columns"%(b2.shape))
#%%
h=sigmoid_activation(torch.mm(train_img,W1) + b1)
y=sigmoid_activation(torch.mm(h,W2) + b2)
print(y)
#%%
prob=softmax(y)
print(prob)
print(torch.sum(prob,dim=1))
#%%
#So far we have just buuld network and not trained.
#Below sets up network in clean way and train that network
class Network(torch.nn.Module):
    def __init__(self,n_input=784,n_hidden=256,n_output=10):
        super().__init__()
        
        #Inputs to Hidden Layer Linear Transformations           
        self.hidden = torch.nn.Linear(n_input,n_hidden)
        
        #Hidden to output Layer transformations
        self.output = torch.nn.Linear(n_hidden,n_output)
        
    def forward(self,x):
        #Hidden with sigmoid activation
        h = torch.sigmoid(self.hidden(x)) 
        #output with softmax
        y = torch.softmax(self.output(h),dim=1)
        return y
#%%
n1 = Network()
y=n1.forward(train_img)   
print(y.shape)     
print(torch.sum(y,dim=1))
#%%
model = torch.nn.Sequential(torch.nn.Linear(n_input,n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden,n_output),
                            )
criterion = torch.nn.CrossEntropyLoss()
#Forward Pass
logits = model(train_img)
loss = criterion(logits,labels)
print(loss)
#%%
print("Before Backward Propogation:{}".format(model[0].weight.grad))
print("Before Backward Propogation:{}".format(model[2].weight.grad))

loss.backward(retain_graph=True)

print("Before Backward Propogation:{}".format(model[2].weight.grad))
print("After Backward Propogation:{}".format(model[0].weight.grad))
#%%
model = torch.nn.Sequential(torch.nn.Linear(n_input,n_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden,n_output),
                            )
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.003)
optimizer.zero_grad()
#Train iteratively
epochs=5

for i in range(epochs):
    running_loss = 0
    for image, label in train_loader:
        train_img=image.view(image.shape[0],-1)
        
        #Forward Pass
        logits = model.forward(train_img)
        loss = criterion(logits,labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        #print("Current Epoch loss:{}".format(loss))
        #print("Running Loss:{}".format(running_loss))
    else:
        print("Training Loss for Epoch {}:{}".format(i,running_loss / (len(train_loader))))
#%%        
