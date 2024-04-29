#importing necessary libraries
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
print("Modules imported")
trainset=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
validset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(trainset,batch_size=64,shuffle=True)
validation_loader=DataLoader(validset,batch_size=64,shuffle=True)
print("Data loaded")
''' To check out an example image
data=iter(train_loader)
images,labels=data.next()
print(images.shape)
print(labels.shape)
plt.imshow(images[0].numpy().squeeze(),cmap='gray_r')
'''
in_size=784
hid_sizes=[128,64]
out_size=10
model=nn.Sequential(nn.Linear(in_size,hid_sizes[0]),nn.ReLU(),nn.Linear(hid_sizes[0],hid_sizes[1]),nn.ReLU(),nn.Linear(hid_sizes[1],out_size),nn.LogSoftmax(dim=1))
criterion=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
t0=time()
epochs=15
for epoch in range(epochs):
    print("Epoch: "+str(epoch+1))
    running_loss=0
    for images, labels in train_loader:
        images=images.view(images.shape[0],-1)
        optimizer.zero_grad()
        z=model(images)
        loss=criterion(z,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
print("Training done")
print("\nTraining time in minutes: "+str((time()-t0)/60))
'''Example validation below'''
images,labels=next(iter(validation_loader))
img=images[0].view(1,784)
with torch.no_grad():
    logps=model(img)
ps=torch.exp(logps)
prob=print(ps.numpy()[0])
print("Predicted Digit = "+str(prob.index(max(prob)))+" with "+str(max(prob)*100)+"% probability")
plt.imshow(images[0].numpy().squeeze(),cmap='gray_r')
plt.show()
