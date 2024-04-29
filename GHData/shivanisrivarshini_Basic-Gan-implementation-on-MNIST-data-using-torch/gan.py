import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from torch.autograd.variable import Variable
import seaborn as sns



def load_training_data():    # Load the dataset
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([.5],[ .5])
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def load_noise_data(size):  # generating random noise sample shape of sample is same as training set shape
    
    noise=Variable(torch.randn(size,100))
    return noise

def Discriminator(): #implement discriminator model
  d_model=nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
           )
   
 
  return d_model

discriminator = Discriminator()

def images_to_vectors(images):
    return images.view(images.size(0), 784)


def Generator(): # implement generator model
    g_model=nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )   
    
    return g_model

generator = Generator()


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, Variable(torch.ones(N,1)) )
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake,Variable(torch.zeros(N,1)))
    error_fake.backward()
    optimizer.step()
    
    
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, Variable(torch.ones(N,1)))
    error.backward()
    optimizer.step()
    return error

data=load_training_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)
noise=load_noise_data(1)
generator=Generator()
discriminator=Discriminator()
num_epochs = 100
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N=real_batch.shape[0]
        real_data=Variable(images_to_vectors(real_batch))
        fake_data=generator(load_noise_data(N)).detach()
        d_error,predict_real,predict_fake=train_discriminator(d_optimizer,real_data,fake_data)
        fake_data=generator(load_noise_data(N))
        g_error=train_generator(g_optimizer,fake_data)
        if(n_batch%100==0):
             test=generator(noise).detach()
             image=test.view(28,28)
             plt.figure(figsize=(3,2))
             plt.imshow(image)
             plt.show()
             print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, n_batch, num_batches))
             print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
             print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(predict_real.mean(), predict_fake.mean()))
        
    
    
    
    




    
    
    

    
    
