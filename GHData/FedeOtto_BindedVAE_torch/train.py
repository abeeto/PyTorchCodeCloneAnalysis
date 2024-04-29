"""
@author: Federico Ottomano
"""

import torch.optim as optim
from data import binded_dataset
from model import BindedVAE
import torch
from utils import loss_maskVAE, loss_ratioVAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader


#Fractional compositions from 'mp-non-metals.csv'
#https://github.com/CompRhys/roost/tree/main/data/datasets/roost

df = pd.read_csv('data.csv')

#%%HYPERPARAMETERS

BATCH_SIZE = 512
N_EPOCHS = 200

INPUT_SIZE = 103
MASK_LATENT = 64
RATIO_LATENT = 32

LEARNING_RATE = 0.001
LAMBDA_M = 10
LAMBDA_R = 1
GAMMA = 10
BETA = 10

#%% INITIALIZATIONS

binded_data = binded_dataset(df)

binded = BindedVAE(INPUT_SIZE, MASK_LATENT, INPUT_SIZE, RATIO_LATENT)

optim_bvae = optim.Adam(binded.parameters(), lr = LEARNING_RATE)

loader_binded = DataLoader(binded_data, batch_size = BATCH_SIZE)

#%% TRAINING BINDED_VAE

for epoch in range(N_EPOCHS):
    
    print(f'Epoch: {epoch + 1} / {N_EPOCHS}')
        
    for i, (masks , ratios) in enumerate(tqdm(loader_binded)):
        
        optim_bvae.zero_grad()
        
        (recon_mask, mu_mask, logvar_mask,
                recon_ratio, mu_ratio, logvar_ratio) = binded(ratios,masks)
        
        
        loss_mask = loss_maskVAE(LAMBDA_M,  
                                 GAMMA, 
                                 masks, 
                                 recon_mask, 
                                 mu_mask, 
                                 logvar_mask)
        
                        
        loss_ratio = loss_ratioVAE(LAMBDA_R,
                                    ratios,
                                    recon_mask,
                                    recon_ratio,
                                    mu_ratio,
                                    logvar_ratio)
        
        loss_binded = BETA*loss_mask + loss_ratio
        
        loss_binded.backward()
                
        optim_bvae.step()
        
    print(f"Total Loss: {loss_binded:.3f}, \t Mask Loss: {loss_mask:.3f}, \t Ratio Loss : {loss_ratio:.3f}")
    
#%%

binded = BindedVAE(INPUT_SIZE, MASK_LATENT, INPUT_SIZE, RATIO_LATENT)

binded.load_state_dict(torch.load('./trained_binded_140ep.pkl'))


generations = binded.generate_materials(40000)
        
def plot_generations(generated_data):
    
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = False
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
    fig.subplots_adjust(hspace=.5)
    im = ax[0].imshow(generated_data, aspect='auto',interpolation='nearest',cmap= plt.get_cmap('viridis'))   
    ax[0].set_title('generation')
    im = ax[1].imshow(binded_data.ratios, aspect='auto',interpolation='nearest',cmap=plt.get_cmap('viridis'))
    ax[1].set_title('real dataset')
    fig.colorbar(im, ax=ax)
    plt.show()
    
plot_generations(generations)

