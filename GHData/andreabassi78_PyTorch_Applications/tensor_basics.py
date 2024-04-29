# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:52:03 2020

@author: Andrea Bassi
"""

import torch

#x = torch.ones(2,2, requires_grad=True)

x = 5.0 * torch.ones(2,2)

x.requires_grad_(True)

x_true = 5.1 *torch.ones(2,2)

print(x)


criterion = torch.nn.MSELoss()

loss = criterion(x, x_true)


#error = torch.mean(x-x_true**2)

print(loss)

#error.backward() is equivalent to error.backward(torch.tensor(1.)) 
loss.backward(torch.tensor(1.)) 

print(x.grad) # this is derror/dx (a tensor beacuse it is calculated for each element of x)





 