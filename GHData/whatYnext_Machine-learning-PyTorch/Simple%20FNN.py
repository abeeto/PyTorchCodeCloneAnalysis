# -*- coding: utf-8 -*-
"""
Implementation of two feedforward neural network
Hightlights:
    1. .mm tensor multiplication
    2. .clone() is copy
    3. .clamp(min,max) is projection to [min,max]
    4. .t() is transpose

@author: mayao
"""

import torch

#torch.cuda.is_available() # Check is gpu is available
dtype = torch.float
#device = torch.device("cpu") # Uncomment this to run on CPU
device = torch.device("cuda:0") 

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

'''
Output: 
99 908.844970703125
199 6.654762268066406
299 0.07194310426712036
399 0.0011684074997901917
499 0.00011185709445271641
'''