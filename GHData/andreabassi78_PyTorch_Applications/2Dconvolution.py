# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:52:09 2021

@author: andrea
"""

import torch
import numpy as np
import time
from torch.nn.functional import conv2d

CHANNELS = 625 
# 25*25, the images must be placed along channels
# the PSFs must be placed as batches

IM_SIZE = 410

PSF_SIZE = 77 # must be odd to have output size == im_size. 

PADDING = PSF_SIZE//2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('\n Using device:', device, '\n')  

im_np = np.random.randn(1, CHANNELS, IM_SIZE, IM_SIZE)
psf_np = np.random.randn(CHANNELS, 1, PSF_SIZE, PSF_SIZE)

t0 = time.time()

# inputs = torch.randn(1, CHANNELS, IM_SIZE, IM_SIZE).to(device = device) 
inputs = torch.from_numpy(im_np).float().to(device = device)

# filters = torch.randn(CHANNELS, 1, PSF_SIZE, PSF_SIZE).to(device = device) 
filters = torch.from_numpy(psf_np).float().to(device = device)

t1 = time.time()
out = conv2d(inputs, filters,
             bias = None,
             stride = 1,
             padding = PADDING,
             groups = CHANNELS
            ).to(device = device) 

print(f'{out.shape = }')
print(f'Elapsed time for CUDA transfer: {t1-t0}')      
print(f'Elapsed time for 2D convolutions: {time.time()-t1}')      

del inputs, filters, out
if torch.cuda.is_available():
    torch.cuda.empty_cache()