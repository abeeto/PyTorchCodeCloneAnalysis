# -*- coding: utf-8 -*-
"""

@author: juika
"""
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
from matplotlib import patches

import seaborn as sns
import pandas as pd
from scipy import stats

import glob

from skimage import exposure
from skimage.transform import resize

from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.utils
import torchvision

# Add pathes for loading modules 
import sys
sys.path.append('Z:\\Ray\\MyCodes\\myFunctionBox')
sys.path.append("Z:\\Ray\\ProcessedData\\DL_UIHC\\_Models")


# Ray's functions
import functionsDataLoader as fDataLoader
import functionscustompltimage as fPlt # functions about colorbar
import parametersettings as paraSet # parameters for overall setting
import functionsms as fMS # for measurements



import vae_2digits_4levels_MSE as mVAE_4 # vae model with 2 latent digits and 4 levels of conv

# Original setting for the cropped image size
output_height, output_width = 162, 162 
cwd = os.getcwd()



output_folder = "Z:\\Ray\\ProcessedData\\DL_UIHC\\2022-09-BasicVAE-MixSet-GCIPL\\basicVAE_epoch_400_mse_1.0_beta_1.25_GCIPL"

model_file = os.path.join(output_folder, "basicVAE_epoch_400_mse_1.0_beta_1.25_GCIPL.pth")

capacity = 64
vae = mVAE_4.VariationalAutoencoder(capacity, False)
vae.load_state_dict(torch.load(model_file))


use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
vae = vae.to(device)


# Call Zeiss color mapping
cmap_z, norm_z = fPlt.create_Zeiss_colorbar()
disp_min, disp_max = 0, 225
max_z_value = display_z_value = disp_max

input_img_range = [0, disp_max] # the scale range; here means the thickness
output_img_size = [112, 112] # quick test
batch_size = 128 # 

clamp_value_high_for_display = 1.2 # after the output from the network, clamp value for further display
img_show_num = 36





## Read dataset
training_dataset ="VIP-Football-GCIPL-Training"
training_df = pd.read_excel("01_Training_Data_List.xlsx", index_col=0)
vip_training_dataset = fDataLoader.Dataset_Image2D(training_df, input_img_range, output_img_size)


train_data_loader = DataLoader(vip_training_dataset, batch_size=batch_size, shuffle=True)
num_total_imgs = len(train_data_loader.dataset)
print("Total images: {}".format(num_total_imgs))



sample = next(iter(train_data_loader))
plt.figure()
img = torchvision.utils.make_grid(sample["image"], 10, 2)
img = img.numpy() # with 3 channels; C x H x W
plt.imshow(max_z_value*(np.transpose(img, (1, 2, 0))[:,:,0]),
            cmap=cmap_z, vmin=disp_min, vmax=disp_max)
plt.axis('off')
plt.clim(0, display_z_value)
plt.title("Training Set - Original")
plt.show()


# Visualization
def visualize_output(images, model, filename=""):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        rec_det, _, _ = model(images)
        
        output_images = rec_det.cpu()
        output_images = output_images.clamp(0,clamp_value_high_for_display)
        np_imagegrid = torchvision.utils.make_grid(output_images[:img_show_num], 6, 2).numpy()
        
        title = filename.split(os.path.sep)[-1]
        plt.figure(figsize=(8,6))
        plt.imshow(input_img_range[1]*(np.transpose(np_imagegrid, (1, 2, 0))[:,:,0]), 
                   cmap=cmap_z, vmin=disp_min, vmax=disp_max)
        plt.axis('off')
        plt.clim([input_img_range[0], display_z_value])
        plt.title(title)
        plt.colorbar()
        plt.show()
        if filename != "":
            plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0.2)



sample = next(iter(train_data_loader))

print("Original Thickness Maps")
plt.figure(figsize=(8,6))
img = torchvision.utils.make_grid(sample["image"][:img_show_num], 6, 2)
img = img.numpy() # with 3 channels; C x H x W
plt.imshow(max_z_value*(np.transpose(img, (1, 2, 0))[:,:,0]),
           cmap=cmap_z, vmin=disp_min, vmax=disp_max)
plt.axis('off')
plt.clim(0, display_z_value)
plt.title("Training Set - Original")
plt.colorbar()
plt.show()
plt.savefig(os.path.join(output_folder,"Trainingset_original_GCIPL_map.png"), bbox_inches='tight', pad_inches=0.2)


print("Trainingset - Basic VAE reconstruction images")
visualize_output(sample["image"], vae, filename=os.path.join(output_folder,"Trainingset_Reconstructed_GCIPL_map.png"))





## Create the latent space scatter plot
label_list = []
latent_list = []
# compute latent variables
vae.eval()
with torch.no_grad():
    for samples in train_data_loader:
        labels = samples["label"]
        images = samples["image"].to(device)
        
        latents, _ = vae.encoder(images)

        label_list = label_list + list(labels)
        latent_list = latent_list + list(latents.to("cpu").numpy().tolist())

df_latent = pd.DataFrame({"Label":label_list, "Latents":latent_list})
df_latent[["d1", "d2"]] = df_latent["Latents"].to_list()
df_latent[["Group", "Dataset", "Subject","Date","Eye","Scan"]] = df_latent["Label"].str.split("_", expand=True)

df_latent.to_excel(os.path.join(output_folder,"02_TrainingSet_Latent_Space.xlsx"))


df_G = df_latent[df_latent["Group"]=="Glaucoma"]
df_N = df_latent[df_latent["Group"]=="Normal"]


fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(x=df_G["d1"], y=df_G["d2"], c='red', marker="o", s=1, label="Glaucoma")
plt.scatter(x=df_N["d1"], y=df_N["d2"], c='blue', marker="o", s=1, label="Normal")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
plt.xlabel('d1')
plt.ylabel('d2')
plt.legend()
fig.savefig(os.path.join(output_folder,"02_Scatter_1.png"), bbox_inches='tight', pad_inches=0.1)






## create the latent space montage map
latent_min_d1, latent_max_d1 = -4, 4
latent_min_d2, latent_max_d2 = -4, 4
tile_width = 0.5 # gap of 0.25 per tile
num_tiles_row = int(1 + (latent_max_d1-latent_min_d1)/tile_width)
num_tiles_total =  num_tiles_row * num_tiles_row
display_adjust = 0.5*tile_width
#extent_setting = [latent_min-display_adjust, latent_max+display_adjust, latent_min-display_adjust, latent_max+display_adjust]
extent_setting = [latent_min_d1-display_adjust, latent_max_d1+display_adjust, latent_min_d2-display_adjust, latent_max_d2+display_adjust]

"""
Show 2D latent space
"""
vae.eval()
with torch.no_grad():
    latent_x = np.linspace(latent_min_d1, latent_max_d1, num_tiles_row) # left: -, right: +
    latent_y = np.linspace(latent_max_d2, latent_min_d2, num_tiles_row) # up: +, down: -
    latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
    
    for j, ly in enumerate(latent_y):
        for i, lx in enumerate(latent_x):
            latents[j,i,0] = lx
            latents[j,i,1] = ly
    latents = latents.view(-1, 2) # flatten grid into a batch
    
    # reconstruct images using the latent vectors
    latents = latents.to(device)
    image_recon = vae.decoder(latents)
    image_recon = image_recon.cpu()
    
    image_display = image_recon.clamp(0,clamp_value_high_for_display)

    
    ## Latent space map
    np_imagegrid = torchvision.utils.make_grid(image_display[:num_tiles_total], num_tiles_row, 2).numpy()
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(np.transpose(input_img_range[1]*np_imagegrid, (1,2,0))[:,:,0], 
               cmap=cmap_z, vmin=disp_min, vmax=disp_max)
    plt.title("2D Latent Vector Space")
    plt.xlabel('d1')
    plt.ylabel('d2')
    plt.axis("off")
        
    plt.show()
    plt.savefig(os.path.join(output_folder,"Trainingset_latent_space_2D_GCIPL_BasicVAE.png"), bbox_inches='tight', pad_inches=0.2)
    

    ## Latent space map with axises
    np_imagegrid_clean = torchvision.utils.make_grid(input_img_range[1]*image_display[:num_tiles_total], num_tiles_row, 0).numpy()
    arr = np.transpose(np_imagegrid_clean, (1,2,0))[:,:,0]
    np.save(os.path.join(output_folder,"Trainingset_latent_space_2D_GCIPL.npy"), arr)
    
    fig_title = "Basic VAE Latent Space - GCIPL"
    fig, ax = plt.subplots(figsize=(10,10))
    cf = ax.imshow(arr, extent=extent_setting,
                   cmap=cmap_z, vmin=disp_min, vmax=disp_max)
    #ax.set_title("2D Latent Sapce - Overall")
    #ax.set_xlabel('d1')
    #ax.set_ylabel('d2')
    grid_tick_x = np.linspace(latent_min_d1, latent_max_d1, num_tiles_row)-display_adjust
    grid_tick_y = np.linspace(latent_min_d2, latent_max_d2, num_tiles_row)-display_adjust
    ax.set_xticks(grid_tick_x, minor=True)
    ax.set_yticks(grid_tick_y, minor=True)
    ax.set_title(fig_title)
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)     
    # ax.grid(which='minor', color="black")   
    cf.set_clim(vmin=disp_min, vmax=disp_max)
    fig.colorbar(cf, shrink=0.8)    
    plt.savefig(os.path.join(output_folder,"Trainingset_latent_space_2D_GCIPL_BasicVAE_Axises.pdf"), bbox_inches='tight', pad_inches=0.05)
      
    

    fig, ax = plt.subplots(figsize=(10,10))
    cf = ax.imshow(arr, extent=extent_setting, cmap=cmap_z, vmin=disp_min, vmax=disp_max, alpha=0.7)
    plt.scatter(x=df_N["d1"], y=df_N["d2"], c='blue', marker="o", s=5, label="Normal")
    plt.scatter(x=df_G["d1"], y=df_G["d2"], c='red', marker="o", s=5, label="Glaucoma")
    grid_tick_x = np.linspace(latent_min_d1, latent_max_d1, num_tiles_row)-display_adjust
    grid_tick_y = np.linspace(latent_min_d2, latent_max_d2, num_tiles_row)-display_adjust
    ax.set_xticks(grid_tick_x, minor=True)
    ax.set_yticks(grid_tick_y, minor=True)
    ax.set_title("Basic VAE Latent Space - GCIPL")
    plt.legend()
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)     
    # ax.grid(which='minor', color="black")   
    cf.set_clim(vmin=disp_min, vmax=disp_max)
    fig.colorbar(cf, shrink=0.8)    
    plt.savefig(os.path.join(output_folder,"Trainingset_latent_space_2D_GCIPL_BasicVAE_Axises_Scatter.pdf"), bbox_inches='tight', pad_inches=0.05)
      
 


## Read dataset
test_dataset ="Test-Set"

if os.path.isfile(os.path.join(output_folder,"02_Test_Data_List.xlsx")):
    test_df = pd.read_excel(os.path.join(output_folder,"02_Test_Data_List.xlsx"), index_col=0)
else:
    data_list = []
    data_folder = "Z:\\Ray\\ProcessedData\\DL_UIHC\\_Data\\VAE_Macula_GCIPL"
    test_folder = ["IowaTrack", "HoodGlaucoma_Set1"]
    for dd in test_folder:
        for root, dirs, files in os.walk(os.path.join(data_folder, dd)):
            for f in files:
                if (f == "crop_GCIPL_thickness_map_162x162.txt") or (f == "Crop_GCIPL_thickness_map_162x162.txt"):
                    strings = root.split(os.path.sep)
                    
                    if dd == "IowaTrack":
                        group = "Normal"
                        team = "IowaTrack"
                        
                    elif dd == "HoodGlaucoma_Set1":
                        group = "Glaucoma"
                        team = "Hood"
                    
                    subject = strings[-4]
                    date = strings[-3]
                    eye = strings[-2]
                    scan = strings[-1]
                    row = {"Group":group, "Subject":subject, "Date":date, "Eye":eye, "Scan":scan,
                           "Label":"_".join([group, team, subject, date, eye, scan]),
                           "Path":os.path.join(root, f)}
                    data_list.append(row)        
    test_df = pd.DataFrame(data_list)      
    test_df.to_excel(os.path.join(output_folder,"02_Test_Data_List.xlsx"))


test_dataset = fDataLoader.Dataset_Image2D(test_df, input_img_range, output_img_size)



test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
num_total_test_imgs = len(test_data_loader.dataset)
print("Total images: {}".format(num_total_test_imgs))






## Test set latent space
label_list = []
latent_list = []
# compute latent variables
vae.eval()
with torch.no_grad():
    for samples in test_data_loader:
        labels = samples["label"]
        images = samples["image"].to(device)
        
        latents, _ = vae.encoder(images)
        

        label_list = label_list + list(labels)
        latent_list = latent_list + list(latents.to("cpu").numpy().tolist())

test_df_latent = pd.DataFrame({"Label":label_list, "Latents":latent_list})
test_df_latent[["d1", "d2"]] = test_df_latent["Latents"].to_list()
test_df_latent[["Group", "Dataset", "Subject","Date","Eye","Scan"]] = test_df_latent["Label"].str.split("_", expand=True)
test_df_latent = test_df_latent.sort_values(by=["Subject", "Date", "Eye", "Scan"])
test_df_latent.to_excel(os.path.join(output_folder,"TestSet_Latents.xlsx"))


test_df_G = test_df_latent[test_df_latent["Group"]=="Glaucoma"]
test_df_N = test_df_latent[test_df_latent["Group"]=="Normal"]
test_df_Track = test_df_latent[test_df_latent["Dataset"] == "IowaTrack"]
test_df_Hood = test_df_latent[test_df_latent["Dataset"]=="Hood"]


fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(x=test_df_G["d1"], y=test_df_G["d2"], c='purple', marker="x", s=15, label="Glaucoma")
plt.scatter(x=test_df_N["d1"], y=test_df_N["d2"], c='black', marker="x", s=15, label="Normal")
ax.set_xlim(latent_min_d1, latent_max_d1)
ax.set_ylim(latent_min_d2, latent_max_d2)
ax.set_aspect('equal')
plt.xlabel('Basic VAE d1')
plt.ylabel('Basic VAE d2')
plt.legend()
fig.savefig(os.path.join(output_folder,"02_Scatter_TestSet.png"), bbox_inches='tight', pad_inches=0.1)



fig_title = "Basic VAE Latent Space - GCIPL - Test Set"
fig, ax = plt.subplots(figsize=(10,10))
cf = ax.imshow(arr, extent=extent_setting,
               cmap=cmap_z, vmin=disp_min, vmax=disp_max, alpha=0.2)
#ax.set_title("2D Latent Sapce - Overall")
#ax.set_xlabel('d1')
#ax.set_ylabel('d2')

plt.scatter(x=test_df_G["d1"], y=test_df_G["d2"], c='purple', marker="x", s=25, label="Glaucoma [DrHood]")
plt.scatter(x=test_df_N["d1"], y=test_df_N["d2"], c='black', marker="x", s=25, label="Normal [IowaTrack]")

plt.scatter(x=df_N["d1"], y=df_N["d2"], c='blue', marker="o", s=1, label="Normal [Training]")
plt.scatter(x=df_G["d1"], y=df_G["d2"], c='red', marker="o", s=1, label="Glaucoma [Training]")

    

grid_tick_x = np.linspace(latent_min_d1, latent_max_d1, num_tiles_row)-display_adjust
grid_tick_y = np.linspace(latent_min_d2, latent_max_d2, num_tiles_row)-display_adjust
ax.set_xticks(grid_tick_x, minor=True)
ax.set_yticks(grid_tick_y, minor=True)
ax.set_title(fig_title)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)     
# ax.grid(which='minor', color="black")   
cf.set_clim(vmin=disp_min, vmax=disp_max)
fig.colorbar(cf, shrink=0.8)    
plt.legend()
plt.savefig(os.path.join(output_folder,"TestSet_latent_space_2D_GCIPL_BasicVAE_Axises.pdf"), bbox_inches='tight', pad_inches=0.05)
  










































