# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:01:06 2021

@author: Andrea Bassi
"""

import torch
import numpy as np
import time
from torch.nn.functional import conv3d
from skimage import color, data
from skimage.transform import resize
import matplotlib.pyplot as plt

def richardson_lucy_3D(signal, kernel, prior=None, iterations=20, verbose=True):
    """
    (Depthwise) Deconvolution using the Richardson Lucy algorithm.

    Parameters
    ----------
    signal: signal to be deblurred
            Torch Tensor with size:  1,Channels,Z,Y,X (Batch=1)
    kernel: Point spread function
            Torch Tensor with size:  Channels,1,Z,Y,X
   
    prior: the initial signal to start the reconstruction. 
           Optional. If not specified the input signal is used as prior  

    Returns
    -------
    signal_deconv : deconvolved signal, same size as signal

    """
    #start_time = time.time()
    
    [B,C,Z,Y,X] = signal.shape
    [_,_,psfZ,psfY,psfX] = kernel.shape
    
    padding = (psfZ//2,psfY//2,psfX//2)
    
    epsilon = 1e-7
    if prior is None: 
        signal_deconv = signal.clone().detach()
        # signal_deconv = torch.rand(B, C, Z, Y, X, device = device) 
    else:
        signal_deconv = prior
        
    kernel_mirror = torch.flip(kernel,[-1,-2,-3]) # flip the Z,Y,X axes
    
    for iteration in range(iterations):
        if verbose==True:
            print(f'{iteration = }')
        
        relative_blur = conv3d(signal_deconv, kernel,
                               bias = None,
                               stride = 1,
                               padding = padding,
                               groups = C
                               )
               
        relative_blur = signal / relative_blur
        
        #avoid errors due to division by zero or inf
        relative_blur[torch.isinf(relative_blur)] = epsilon
        relative_blur[torch.isnan(relative_blur)] = 0
        relative_blur = torch.abs(relative_blur)

        # multiplicative update 
        signal_deconv *= conv3d(relative_blur, kernel_mirror,
                                bias = None,
                                stride = 1,
                                padding = padding,
                                groups = C
                                )
        
    return signal_deconv


def gaussian(Lx,Ly,Lz, Wx, Wy, Wz, X0=0, Y0=0, Z0=0):
    """ creates a 3D gaussian (numpy array with size Lz,Ly,Lx)
    with the specified waist along x, y and z """
    
    x_lin = np.linspace(-Lx/2, Lx/2, Lx)
    y_lin = np.linspace(-Ly/2, Ly/2, Ly)
    z_lin = np.linspace(-Lz/2, Lz/2, Lz)
    
    Z_np, Y_np, X_np = np.meshgrid(z_lin,y_lin,x_lin, indexing ='ij')

    kern = 1.0 * np.exp( - ((X_np-X0)**2)/Wx**2 
                         - ((Y_np-Y0)**2)/Wy**2
                         - ((Z_np-Z0)**2)/Wz**2
                         )
    kern = kern/np.sum(kern)
    return kern, X_np, Y_np, Z_np


if __name__ == '__main__':

    BATCH = 1 # number of input datasets
    CHANNELS = 2 # note that according to torch notation we have in_channels = out_channels
    # here we are doing "depthwise" convolutions: group=in_channels
    IM_SIZE = 100 
    Z_SIZE = 100
    PSF_XYSIZE = 15 # must be odd to have output size == im_size. 
    PSF_ZSIZE = 15 # must be odd to have output size == im_size. 
    
    PADDING = (PSF_ZSIZE//2, PSF_XYSIZE//2, PSF_XYSIZE//2)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('\n Using device:', device, '\n') 


    # generate a 3D image with size: BATCH, CHANNELS, Z_SIZE, IM_SIZE, IM_SIZE   
    im_np = resize(color.rgb2gray(data.astronaut()), [IM_SIZE,IM_SIZE], mode='constant')
    im_np = im_np /np.amax(im_np)
    im_np = np.tile(im_np,(BATCH,CHANNELS,Z_SIZE,1,1))
    #im_np[0,0,0,10,10] = 20      
    
    # generate a 3D PSF with size: CHANNELS, 1, PSF_ZSIZE, PSF_XYSIZE, PSF_XYSIZE   
    Wx = 0.5 # pixel unit
    Wy = 3.5 # pixel unit
    Wz = 0.5 # pixel unit
    X0 = 2.0
    Y0 = 1.0
    Z0 = 0
    psf_np, _, _, _ = gaussian(PSF_XYSIZE, PSF_XYSIZE, PSF_ZSIZE, Wx, Wy, Wz, X0, Y0, Z0)
    psf_np = np.tile(psf_np,(CHANNELS,1,1,1,1))
    #psf_np[0,0,0,-1,-1] = 10 
    
          
    
    #im_np += np.random.randn(1, CHANNELS, IM_SIZE, IM_SIZE)
    #psf_np += np.random.randn(CHANNELS, 1, PSF_SIZE, PSF_SIZE)
    
    print(f'{psf_np.shape = }')
    print(f'{im_np.shape = }')
    
    t0 = time.time()
    
    # inputs = torch.randn(1, CHANNELS, IM_SIZE, IM_SIZE).to(device = device) 
    inputs = torch.from_numpy(im_np).float().to(device = device)
    
    # filters = torch.randn(CHANNELS, 1, PSF_SIZE, PSF_SIZE).to(device = device) 
    psf = torch.from_numpy(psf_np).float().to(device = device)
    
    t1 = time.time()
    out = conv3d(inputs, psf,
                 bias = None,
                 stride = 1,
                 padding = PADDING,
                 groups = CHANNELS
                ).to(device = device)
    print(f'{out.shape = }')  
    
    t2 = time.time()
    restored = richardson_lucy_3D(out, psf, iterations=10)
    
    print(f'Elapsed time for CUDA transfer: {t1-t0}')      
    print(f'Elapsed time for convolution: {t2-t1}') 
    print(f'Elapsed time for deconvolution: {time.time()-t2}')      
    
    
      
    plt.figure(figsize=(6, 6))
    plt.gray()
    plt.imshow(im_np[0,0,0,:,:])
    plt.title('Original image')
    plt.colorbar()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(psf_np[0,0,0,:,:])
    plt.title('Gaussian point spread function')
    plt.colorbar()
    
    blurred_to_show = out[0,0,0,:,:]
    plt.figure(figsize=(6, 6))
    plt.imshow(blurred_to_show.cpu().numpy())
    plt.title('Blurred image');
    plt.colorbar()
    
    restored_to_show = restored[0,0,0,:,:]
    plt.figure(figsize=(6, 6))
    plt.imshow(restored_to_show.cpu().numpy())
    #plt.imshow(restored_to_show.cpu().numpy(),vmin=0,vmax=5)
    plt.title('Restored image');
    plt.colorbar()
    
    print(f'{out.shape = }')
         
    del inputs, psf, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()