# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:54:13 2020

@author: Andrea Bassi
"""

import torch
from skimage import color, data
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import numpy as np
import time

dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('\n Using device:', device, '\n')  

K_SIZE = 23 # PSF size
PADDING = K_SIZE//2
IM_SIZE = 200# image size

im_np = resize(color.rgb2gray(data.astronaut()), [IM_SIZE,IM_SIZE], mode='constant')
im_np = im_np /np.amax(im_np)

plt.figure(figsize=(6, 6))
plt.gray()
plt.imshow(im_np)
plt.title('Original image')
plt.colorbar()

def gaussian_kernel(kernlen, Wx, Wy, X0=0, Y0=0):
    x_lin = y_lin = np.linspace(-K_SIZE/2, K_SIZE/2, K_SIZE)
    X_np, Y_np = np.meshgrid(x_lin,y_lin)
    kern = 1.0 * np.exp( - ((X_np-X0)**2)/Wx**2 
                         - ((Y_np-Y0)**2)/Wy**2 )
    kern = kern/np.sum(kern)
    return kern, X_np, Y_np

Wx = 3. # pixel unit
Wy = 1. # pixel unit
X0 = 6.
Y0 = 3.

psf_np, _, _ = gaussian_kernel(K_SIZE, Wx, Wy, X0, Y0)

#psf_np[-5,-5] = 1 
#psf_np[5,-5] = 1

print('PSF:',
      '\n --> Amplitude =', np.amax(psf_np),
      '\n --> Waist x =', Wx,
      '\n --> Waist y =', Wy,
      '\n')

plt.figure(figsize=(6, 6))
plt.imshow(psf_np)
plt.title('Gaussian point spread function');
plt.colorbar()
plt.pause(0.05)


im = torch.from_numpy(im_np).float().to(device = device) 
psf = torch.from_numpy(psf_np).float().to(device = device) 

#print('im shape:', im.shape)

im = im.expand([1,1,IM_SIZE,IM_SIZE]) 
#don't know why, but I was not able to do the convolution without adding 2 channels with 1 single element 
psf = psf.expand([1,1,K_SIZE,K_SIZE])

#generate blurred images with torch.nn.functional.conv2d
im_blurred = torch.nn.functional.conv2d(im, psf, bias=None, stride=1, padding=PADDING, dilation=1, groups=1).to(device = device)  

#add noise
#im_blurred += 0.01* torch.randn(im_blurred.shape).to(device = device) 

#print('\n im_blurred + noise mean:')
#print(torch.mean(im_blurred))

plt.figure(figsize=(6, 6))
plt.imshow(im_blurred.squeeze().cpu()) #opposite of expand
plt.title('Blurred image');
plt.colorbar()
plt.pause(0.05)


"""
                        Deconvolution starts here

"""
class Net(torch.nn.Module):

    def __init__(self, im_in):
        super(Net,self).__init__()
        self.im = torch.nn.Parameter(im_in.clone()).to(device = device)
        self.im.requires_grad = True
        
    def forward(self, psf):
        # different from a standard Conv2d layer because in_channel and kernel are inverted
        # it also differs from ConvTranspose2d
        im_pred = torch.nn.functional.conv2d(self.im,
                                             psf,
                                             bias=None,
                                             stride=1,
                                             padding=PADDING,
                                             dilation=1, groups=1
                                             ).to(device = device)
        return im_pred 

initial_guess = im_blurred.clone().detach()
    
net = Net(initial_guess)   


# Define a loss function
loss_fn = torch.nn.MSELoss(reduction='sum').to(device = device)
# loss_fn = torch.nn.SmoothL1Loss()

# Use the optim package to define an Optimizer that will update the weights of the model for us. 
#optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
#optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, weight_decay=0.01)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.05, weight_decay=0.01)

t0= time.time()

for t in range(500):
    '''
    Forward pass: compute predicted blurred image by passing the psf.
    The reconstructed image is stored inside the net and can be called in one 
    of the two following ways:
        net[0].weight
        list(net.parameters())[0].data
    The net has also a bias (of size 1):
        net[1].bias
        list(net.parameters())[1].data
    '''    
    im_pred = net.forward(psf)
    
    # Compute and print loss.
    loss = loss_fn(im_pred, im_blurred)#+loss_fn2(im_pred, im_blurred)
        
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    
    
    if t % 100 == 99: #print loss and show image only sometimes
        
        print('step:',t,
              ',loss:', loss.item()
             )
        
        # plt.figure(figsize=(6, 6))
        # plt.title('Deconvolved image, step:' + str(t));  
        # plt.imshow(net.im.detach().squeeze().cpu().numpy(),
        #             vmin=0,
        #             vmax=1)
        # plt.pause(0.05)

restored = net.im.detach().squeeze().cpu().numpy()

#restored = list(net.parameters())[0].detach().squeeze().cpu().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(restored)
plt.title('Deconvolved (optimization)');
plt.colorbar()

print('\nElapsed time:', time.time()-t0)    

#del net,im,im_blurred,im_pred
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
    
