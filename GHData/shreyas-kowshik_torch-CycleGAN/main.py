import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import cv2
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
from unet import UNet
from net import Discriminator
from tqdm import tqdm

from utils import *
import utils
from data_utils import *

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 1
rec_wt = 0.5
num_epochs = 100
num_examples = 10 # To save time

# Define the transforms
tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

# Create the dataset variables
X_A_dataset = ImageDataset(path='data/trainA/',num_examples=num_examples,transforms=tf)
X_B_dataset = ImageDataset(path='data/trainB/',num_examples=num_examples,transforms=tf)

dataset = MyDataset(X_A_dataset,X_B_dataset)

# Create data loader
train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

D_A = Discriminator().to(device)
G_A = UNet(n_channels=3,n_classes=3).to(device)
D_B = Discriminator().to(device)
G_B = UNet(n_channels=3,n_classes=3).to(device)

criterion = nn.MSELoss()
rec_criterion = nn.L1Loss()
dA_optimizer = torch.optim.Adam(D_A.parameters(), lr=0.0002)
gA_optimizer = torch.optim.Adam(G_A.parameters(), lr=0.0002)
dB_optimizer = torch.optim.Adam(D_B.parameters(), lr=0.0002)
gB_optimizer = torch.optim.Adam(G_B.parameters(), lr=0.0002)

def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

def reset_grad():
        dA_optimizer.zero_grad()
        gA_optimizer.zero_grad()
        dB_optimizer.zero_grad()
        gB_optimizer.zero_grad()

# Start training
total_step = len(train_loader)
for epoch in tqdm(range(num_epochs)):
        for i, (A,B) in tqdm(enumerate(train_loader)):
                # print(i)

                A = A.to(device)
                B = B.to(device)

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                B_fake = G_A(A)
                A_fake = G_B(B)

                A_rec = G_B(B_fake)
                B_rec = G_A(A_fake)

                DA_real = D_A(A).squeeze().to(device)
                DA_fake = D_A(A_fake).squeeze().to(device)

                DB_real = D_B(B).squeeze().to(device)
                DB_fake = D_B(B_fake).squeeze().to(device)

                DA_loss_real = criterion(DA_real,real_labels)
                DA_loss_fake = criterion(DA_fake,fake_labels)
                DA_loss = DA_loss_real + DA_loss_fake

                DB_loss_real = criterion(DB_real,real_labels)
                DB_loss_fake = criterion(DB_fake,fake_labels)
                DB_loss = DB_loss_real + DB_loss_fake

                cycle_loss = rec_criterion(A_rec,A) + rec_criterion(B_rec,B)

                GA_loss = criterion(DA_fake,real_labels) + rec_wt*cycle_loss
                GB_loss = criterion(DB_fake,real_labels) + rec_wt*cycle_loss

                reset_grad()
                DA_loss.backward(retain_graph=True)
                dA_optimizer.step()

                reset_grad()
                GA_loss.backward(retain_graph=True)
                gA_optimizer.step()

                reset_grad()
                DB_loss.backward(retain_graph=True)
                dB_optimizer.step()

                reset_grad()
                GB_loss.backward()
                gB_optimizer.step()

        save_image(denorm(B_fake), 'outputs/fake_B-{}.png'.format(epoch+1))
        save_image(denorm(A_fake), 'outputs/fake_A-{}.png'.format(epoch+1))     
        save(D_A,'Dis_A.ckpt','models/')
        save(D_B,'Dis_B.ckpt','models/')
        save(G_A,'Gen_A.ckpt','models/')
        save(G_B,'Gen_B.ckpt','models/')