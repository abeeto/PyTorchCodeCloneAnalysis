#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:09:38 2019

@author: said
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable


BATCH_SIZE = 128
EPOCHS = 100
ZDIMS = 500
LOG_INTERVAL = 10

trainset = datasets.MNIST('./data/', download=True, train=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                          shuffle=True, num_workers=2)

testset = datasets.MNIST('./data/', download=False,train=False, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, 
                                         shuffle=True, num_workers=2)

#def show_images(images):
#    images = torchvision.utils.make_grid(images)
#    show_image(images[0])
#
#def show_image(img):
#    plt.imshow(img, cmap='gray')
#    plt.show()
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#show_images(images)

device = torch.device("cpu")

class VAE(nn.Module):
    def __init__(self, latent_variable_dim):
        super(VAE, self).__init__()
        #ENCODER
        self.fc1 = nn.Linear(784,400)
        self.relu = nn.ReLU()
        self.fc2m = nn.Linear(400, latent_variable_dim)
        self.fc2s = nn.Linear(400, latent_variable_dim)
        #DECODER
        self.fc3 = nn.Linear(latent_variable_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x:Variable) -> (Variable, Variable):
        h1 = self.relu(self.fc1(x))
        return self.fc2m(h1), self.fc2s(h1)
    
    def reparameterize(self, mu: Variable, log_var: Variable)->Variable:
        if self.training:
            s = log_var.mul(0.5).exp_() #torch.exp(0.5*log_var)
            eps = torch.randn_like(s)
            return eps.mul(s).add_(mu)
        else:
            return mu
    
    def forward(self, input):
        mu, log_var = self.encode(input.view(-1,784))
        z = self.reparameterize(mu,log_var)
        return self.decode(z), mu, log_var
    
    def decode(self,z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    


vae = VAE(ZDIMS) 

def loss_function(recon_image, input_image,mu, log_var)->Variable:
    CE = F.binary_cross_entropy(recon_image, input_image.view(-1,784))
    KLD = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
    KLD /= BATCH_SIZE*784
    return KLD + CE
   
optimizer = optim.Adam(vae.parameters(), lr=0.001)
    

train_loss = []

def train(epoch):
    vae.train()
    train_loss = 0
    
    for batch_idx, (data,_) in enumerate(trainloader):
        data = Variable(data)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss +=loss.item()
        optimizer.step()
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(trainloader.dataset),
                    100. * batch_idx/len(trainloader),
                    loss.item()/len(data)))
    
    print('====> Epoch: {} Average Loss: {:.4f}'.format(
            epoch, train_loss/len(trainloader.dataset)))
    
for epoch in range(1,EPOCHS+1):
    train(epoch)
    
    sample = Variable(torch.randn(64, ZDIMS))
    sample = vae.decode(sample).cpu()
    
    save_image(sample.data.view(64,1,28,28),
               'results/sample_'+str(epoch)+'.png')
     
    
    
#for epoch in range(5):
#    for i,data in enumerate(trainloader,0):
#        images,labels = data
#        images = images.to(device)
#        optimizer.zero_grad()
#        recon_image, s , mu = vae(images)
#        l = vae.loss(images, recon_image, mu,s)
#        l.backward()
#        train_loss.append(l.item()/len(images))
#        optimizer.step()
#
#plt.plot(train_loss)
#plt.show()    