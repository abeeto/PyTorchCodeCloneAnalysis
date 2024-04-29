# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:57:21 2020

@author: juika-Cindy
"""



import os
import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors

from skimage.transform import resize

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.utils as vutils


# Add pathes for loading modules 
import sys


# Ray's functions
import functionsDataLoader as fDataLoader   # functions for dataloader


# import vae_2digits_4levels as mVAE_4 # vae model with 2 latent digits and 4 levels of conv
import vae_2digits_4levels_MSE as mVAE_4 # vae model with 2 latent digits and 4 levels of conv



cwd = os.getcwd()
torch.cuda.empty_cache()

# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 2
num_epochs = 100
batch_size = 128 
capacity = 64
learning_rate = 2e-4 #1e-3



# When beta is high; the VAE will focus in the normalization part intead of making
# the reconstrcuted images to be alike to the input image
mse_coff = 1.0
variational_beta = 1.25# Original: 1.5; # 200 is too high
recon_loss_type = "mse"


use_gpu = True

## Save the trained parameters
model_name = "basicVAE_epoch_{}_mse_{}_beta_{}.pth".format(str(num_epochs), str(mse_coff), str(variational_beta))



disp_min, disp_max = 0, 1
max_z_value = disp_max

input_img_range = [0, disp_max] # the scale range; here means the thickness
output_img_size = [112, 112] # quick test



## Prepare a datafram for the training images
data_list = []
data_folder = "Z:\\GarvinLabDL_Data\\REFUGE\\Mix\\data"
for root, dirs, files in os.walk(data_folder):
    for f in files:
        if "Cropped_Disc.png" in f: 
            strings = f.split("_")
            dataset = strings[0]
            temp_str = strings[1]
            group = temp_str[0]
            imgID = temp_str[1:]
            
            row = {"Dataset":dataset, "Group":group, "ImageID":imgID,
                   "Label":"_".join([dataset, group, imgID]),
                   "Path":os.path.join(root, f)}
            data_list.append(row)        
df = pd.DataFrame(data_list)
df.to_excel("01_Training_Data_List.xlsx")


num_total_scans = len(df)


with open("01_Training_Dataset_Summary.txt", "w") as fff:
    fff.write(f"Total scans: {num_total_scans} \n")



# in the data loader, these images are already tensor images
data_augmentation = transforms.Compose([
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5)])


## Read dataset
image_dataset_name ="REFUGE"
data = fDataLoader.DatasetREFUGE(df, input_img_range, output_img_size)

data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
num_total_imgs = len(data_loader.dataset)
print("Total images: {}".format(num_total_imgs))


# Test
sample = next(iter(data_loader))

plt.figure()
img = torchvision.utils.make_grid(sample["image"], 10, 2)
img = img.numpy() # with 3 channels; C x H x W
plt.imshow(max_z_value*(np.transpose(img, (1, 2, 0))[:,:,0]),
           cmap="gray", vmin=disp_min, vmax=disp_max)
plt.axis('off')
plt.title("Original")
plt.show()




seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



# Training mode: True
vae = mVAE_4.VariationalAutoencoder(capacity, True)


device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
vae = vae.to(device)


num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print("Number of parameters: {}".format(num_params))



"""
# set training mode
"""
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)
vae.train()
train_loss_avg = []

sum_loss_list = []
rec_loss_list = []
KL_loss_list = []


print("Start to train: .....")
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    
    
    batch_sum_loss = 0
    batch_rec_loss = 0
    batch_KL_loss = 0
    
    count = 0
    for sample in data_loader:
        image_batch = sample["image"]
        image_batch = image_batch.to(device)
        b_size = image_batch.size(0)
        

        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

        # reconstruction error
        true_size = output_img_size[0] * output_img_size[1]
        

        rec_loss = mVAE_4.calc_reconstruction_loss(image_batch, image_batch_recon, loss_type=recon_loss_type, reduction="mean")
        KL_loss = mVAE_4.calc_kl(latent_logvar, latent_mu, reduce="mean")
        loss = mse_coff*rec_loss + variational_beta*KL_loss
        
        batch_sum_loss = batch_sum_loss + loss.data.cpu().item()
        batch_rec_loss = batch_rec_loss + rec_loss.data.cpu().item()
        batch_KL_loss = batch_KL_loss + KL_loss.data.cpu().item()

        # backpropogation
        optimizer.zero_grad()
        loss.backward()

        # one step of the optimizer (using the gradients from backpropogation)
        optimizer.step()
        count += 1

    sum_loss_list.append(batch_sum_loss)
    rec_loss_list.append(batch_rec_loss)
    KL_loss_list.append(batch_KL_loss)

    print("Epoch [{}/{}] Reconstruction Loss + KL Loss: {:.3f}".format(epoch, num_epochs, sum_loss_list[-1]))
    print("   Reconstruction Loss: {:.3f}".format(rec_loss_list[-1]))
    print("   KL Loss: {:.3f}".format(KL_loss_list[-1]))



loss_df = pd.DataFrame({"SumLoss":sum_loss_list,
                        "ReconstructionLoss":rec_loss_list,
                        "KL-Loss":KL_loss_list})

loss_df.to_excel("01_Loss.xlsx")



model_file = os.path.join(cwd, model_name)
torch.save(vae.state_dict(), model_file)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(len(rec_loss_list)), rec_loss_list, label="Rec-Loss")
ax.plot(np.arange(len(KL_loss_list)), KL_loss_list, label="KL-Loss")
ax.plot(np.arange(len(sum_loss_list)), sum_loss_list, label="Sum-Loss")
#ax.set_ylim([0, 200])
ax.legend()
plt.yscale("log")
plt.show()
plt.savefig("BasicVAE_Loss_{}.png".format(model_name))
























