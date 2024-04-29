# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:04:17 2020

@author: Andrea Bassi

Fits a 2D gaussian function with added noise to retrieve its amplitude, waist,
and a baseline level.

Solves an inverse problem using a commonly used optimizer and loss function. 
It is implemented using a custumized layer of Pytorch. 

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
device = torch.device("cpu")
print('\n Using device:', device, '\n')    

# Create a 2D gaussian function with noise
IM_SIZE = 20

x_lin = y_lin = np.linspace(-IM_SIZE/2, IM_SIZE/2, IM_SIZE) 
X_np, Y_np = np.meshgrid(x_lin,y_lin)
X0 = 0
Y0 = 0

A = 1 # amplitude
W = 4 # waist (in pixels unit)
B = 0.2 # bias/baseline
N = 0.4 # noise 

def print_values(values, title=''):
    print('\n', title, 'values:'
              '\n --> Amplitude =', values[0],
              '\n --> Waist =', values[1],
              '\n --> Bias =', values[2], 
              '\n'
              )

print_values([A,W,B], 'True')

# generate the true gaussian function 
Z_np = A * np.exp((- (X_np-X0)**2 - (Y_np-Y0)**2)/W**2)
# add noise with uniform distribution between -N/2 and N/2
Z_np += B + N * (np.random.random(size = Z_np.shape)-0.5)

x = torch.from_numpy(X_np).float().to(device = device)
y = torch.from_numpy(Y_np).float().to(device = device)
z = torch.from_numpy(Z_np).float().to(device = device) 

"""
                         Define a net
                         
Here I don't really exploit the layers of Pytorch, I define a custom layer with
parameters self.val that are optimized using the gaussian forward 

"""

class Net(torch.nn.Module):

    def __init__(self, val_in):
        super(Net,self).__init__()
        self.val = torch.nn.Parameter(val_in.clone()).to(device = device)
        self.val.requires_grad = True
        #self.relu = torch.nn.ReLU()
        
    def forward(self, X, Y):
        amp = self.val[0]
        waist = self.val[1]
        bias = self.val[2]
        X0 = 0
        Y0 = 0
        z_pred = ( bias + amp * torch.exp((- (X-X0)**2 - (Y-Y0)**2)/waist**2)
                  ).to(device = device)
        return z_pred 
    
    def print_net_values(self, title = ''):
        print_values(self.val.detach().cpu().numpy(), title)
        
# use reasonable initialization values 
amplitude_guess = np.amax(Z_np)
waist_guess = IM_SIZE*np.std(Z_np)
bkg_guess =  np.mean(Z_np)
 
initial_guess = torch.tensor([amplitude_guess,
                              waist_guess,
                              bkg_guess]
                             ).to(device = device)

# use random initialization values
#initial_guess = torch.rand([3]).to(device = device)

net = Net(initial_guess)

net.print_net_values('Initialization')

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=0.1)

"""

                     Start the optimization

"""
for t in range(50):
    
    # Forward pass: compute predicted y by passing x to the model.
    # This is equivalent to net.forward(x,y)
    z_pred = net(x,y)

    # Compute and print loss.
    loss = loss_fn(z_pred, z)
    if t % 5 == 4:
        
        print('step:',t,
              ',loss:', loss.item()
             )
        

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
net.print_net_values('Fitted')    


def show_image(ax, data, title='', vmin=0, vmax=1):
    ax.imshow(data, interpolation='none',
              cmap='gray',
              origin='lower',
              extent=[-IM_SIZE/2, IM_SIZE/2, -IM_SIZE/2, IM_SIZE/2],
              vmax=vmax, vmin=vmin,
              )
    #ax.axis('equal')
    ax.set(xlabel = 'x',
           ylabel = 'y',
           title = title,
           )

    
input_data = z.detach().cpu().numpy()
output_data = z_pred.detach().cpu().numpy()
difference = output_data-input_data

_fig, axs = plt.subplots(1,3,figsize=(12, 6))
show_image(axs[0],input_data,title ='Original',
           vmin=0, vmax=input_data.max())
show_image(axs[1],output_data,title='Predicted',
           vmin=0, vmax=input_data.max())
show_image(axs[2],difference,title='Difference',
           vmin=difference.min(), vmax=difference.max())


_fig, ax = plt.subplots(figsize=(6, 6))
center_idx = int(input_data.shape[0]/2)
ax.plot(x_lin, input_data[center_idx,:], 'bo', label = ('Original'))
ax.plot(x_lin, output_data[center_idx,:], 'r-', label = ('Predicted'))
ax.legend(loc='upper left', frameon=False)
ax.grid()
ax.set(xlabel = 'x',
       ylabel = 'z',
       xlim = (-IM_SIZE/2, IM_SIZE/2),
       title = 'Predicted vs Original at y = 0'
       )