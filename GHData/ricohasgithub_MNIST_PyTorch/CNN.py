import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

class CNN (nn.Module):
    def __init__(self,out_1=16,out_2=32):
        
        super(CNN,self).__init__()
        #first Convolutional layers
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=out_1,kernel_size=5,padding=2)
        #activation function 
        self.relu1=nn.ReLU()
        #max pooling 
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        #second Convolutional layers
        self.cnn2=nn.Conv2d(in_channels=out_1,out_channels=out_2,kernel_size=5,stride=1,padding=2)
        #activation function 
        self.relu2=nn.ReLU()
        #max pooling 
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        #fully connected layer 
        self.fc1=nn.Linear(out_2*7*7,10)
        
    def forward(self,x):
        #first Convolutional layers
        out=self.cnn1(x)
        #activation function 
        out=self.relu1(out)
        #max pooling 
        out=self.maxpool1(out)
        #first Convolutional layers
        out=self.cnn2(out)
        #activation function
        out=self.relu2(out)
        #max pooling
        out=self.maxpool2(out)
        #flatten output 
        out=out.view(out.size(0),-1)
        #fully connected layer
        out=self.fc1(out)
        
        return out
    
    def activations(self,x):
        #outputs activation this is not necessary just for fun 
        z1=self.cnn1(x)
        a1=self.relu1(z1)
        out=self.maxpool1(a1)
        
        z2=self.cnn2(out)
        a2=self.relu2(z2)
        out=self.maxpool2(a2)
        out=out.view(out.size(0),-1)
        return z1,a1,z2,a2,out
