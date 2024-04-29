'''
Created on Fri Apr 26 18:45:20 2020

@author: Andrea Bassi


Estimates the PSF given one blurred image and the corresponging ground truth image.
It assumes that the PSF is a 2D gaussian function

'''

import torch
from skimage import color, data
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('\n Using device:', device, '\n')  

K_SIZE = 13 # PSF size
PADDING = 3
IM_SIZE = 171# image size

im_np = resize(color.rgb2gray(data.astronaut()), [IM_SIZE,IM_SIZE], mode='constant')
im_np = im_np /np.amax(im_np)

plt.figure(figsize=(6, 6))
plt.gray()
plt.imshow(im_np)
plt.title('Original image')

def gaussian(kernlen, A, Wx, Wy):
    x_lin = y_lin = np.linspace(-K_SIZE/2, K_SIZE/2, K_SIZE)
    X_np, Y_np = np.meshgrid(x_lin,y_lin)
    X0 = 0
    Y0 = 0
    kern = A * np.exp( - ((X_np-X0)**2)/Wx**2 
                       - ((Y_np-Y0)**2)/Wy**2 )
    return kern, X_np, Y_np

Wx = 5. # pixel unit
Wy = 2. # pixel unit
A = 1 # amplitude 

psf_np, X_np, Y_np = gaussian(K_SIZE, A, Wx, Wy)

def print_values(values, title=''):
    print('\n', title,
          '\n --> Amplitude =', values[0],
          '\n --> Waist x =', values[1],
          '\n --> Waist y =', values[2],
          '\n'
          )
#print_values([A, Wx, Wy], 'True values:')

x = torch.from_numpy(X_np).float().to(device = device)
y = torch.from_numpy(Y_np).float().to(device = device)
psf = torch.from_numpy(psf_np).float().to(device = device) 

im = torch.from_numpy(im_np).float().to(device = device) 

im = im.expand([1,1,IM_SIZE,IM_SIZE]) 
#don't know why, but I was not able to do the convolution without adding 2 channels with 1 single element 
psf = psf.expand([1,1,K_SIZE,K_SIZE])

#add noise to psf
psf += 0.1 * torch.randn(psf.shape).to(device = device) 

#generate blurred images with torch.nn.functional.conv2d
im_blurred = torch.nn.functional.conv2d(im, psf,
                                        bias=None, stride=1, padding=PADDING,
                                        dilation=1, groups=1
                                        ).to(device = device)  
#add noise to the blurred image
im_blurred += 0.1* torch.randn(im_blurred.shape).to(device = device) 

plt.figure(figsize=(6, 6))
plt.imshow(im_blurred.squeeze().cpu()) #contrario di expand
plt.title('Blurred image');


class Net(torch.nn.Module):

    def __init__(self, val_in):
        super(Net,self).__init__()
        self.val = torch.nn.Parameter(val_in.clone()).to(device = device)
        self.val.requires_grad = True
        #self.relu = torch.nn.ReLU()
        
    def forward(self, X, Y, im):
        amp = self.val[0]
        waistx = self.val[1]
        waisty = self.val[2]
        X0 = 0
        Y0 = 0
        
        psf = ( amp * torch.exp( - ((X-X0)**2)/waistx**2 
                                 - ((Y-Y0)**2)/waisty**2 )                                      
               ).to(device = device)
        
        self.psf_pred = psf.expand([1,1,K_SIZE,K_SIZE])
        
        im_pred = torch.nn.functional.conv2d(im,
                                             self.psf_pred,
                                             bias=None,
                                             stride=1,
                                             padding=PADDING,
                                             dilation=1, groups=1
                                             ).to(device = device)
        return im_pred
    
    def print_net_values(self, title = ''):
        print_values(self.val.detach().cpu().numpy(), title)
        
# use reasonable initialization values 
amplitude_guess = 0.5
waistx_guess = 1.
waisty_guess = 1.
 
initial_guess = torch.tensor([amplitude_guess,
                              waistx_guess,
                              waisty_guess]
                             ).to(device = device)

# use random initialization values
#initial_guess = torch.rand([4]).to(device = device)

net = Net(initial_guess)

net.print_net_values('Initialization values:')

loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.SmoothL1Loss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.01)
#optimizer = torch.optim.ASGD(model.parameters(), lr=1e-6)

         

for t in range(1500):
    
    # Forward pass: compute predicted y by passing x to the model.
    # This is equivalent to net.forward(x,y)
    im_pred = net.forward(x,y,im)

    # Compute and print loss.
    loss = loss_fn(im_pred, im_blurred)
    if t % 100 == 99:
        
        print('\r', 'step:',t,
              ',loss:', loss.item(), end = ''
             )

        # psf_pred = net.psf_pred.squeeze().detach().cpu().numpy()
        # plt.figure(figsize=(6, 6))
        # plt.title('Predicted PSF, step:' +str(t));    
        # plt.imshow(psf_pred)
        # plt.pause(0.01)

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

"""

                Print and show the data
                
"""
print()

print_values([A,Wx,Wy], 'True values:')

net.print_net_values('Fitted values:')    

def show_image(ax, data, title='', vmin=0, vmax=1):
    ax.imshow(data, interpolation='none',
              cmap='gray',
              origin='lower',
              extent=[-K_SIZE/2, K_SIZE/2, -K_SIZE/2, K_SIZE/2],
              vmax=vmax, vmin=vmin,
              )
    #ax.axis('equal')
    ax.set(xlabel = 'x',
           ylabel = 'y',
           title = title,
           )      

input_data  = psf.squeeze().detach().cpu().numpy()
output_data = net.psf_pred.squeeze().detach().cpu().numpy()
difference  = output_data-input_data

_fig, axs = plt.subplots(1,3,figsize=(12, 6))
show_image(axs[0],input_data,title ='True',
           vmin=0, vmax=input_data.max())
show_image(axs[1],output_data,title='Predicted',
           vmin=0, vmax=input_data.max())
show_image(axs[2],difference,title='Difference',
           vmin=difference.min(), vmax=difference.max())

x_lin = np.linspace(-K_SIZE/2, K_SIZE/2, K_SIZE)

_fig, ax = plt.subplots(figsize=(6, 6))
center_idx = int(input_data.shape[0]/2)
ax.plot(x_lin, input_data[center_idx,:], 'bo', label = ('Original'))
ax.plot(x_lin, output_data[center_idx,:], 'r-', label = ('Predicted'))
ax.legend(loc='upper left', frameon=False)
ax.grid()
ax.set(xlabel = 'x',
       ylabel = 'z',
       xlim = (-K_SIZE/2, K_SIZE/2),
       title = 'Predicted vs Original at y = 0'
       )