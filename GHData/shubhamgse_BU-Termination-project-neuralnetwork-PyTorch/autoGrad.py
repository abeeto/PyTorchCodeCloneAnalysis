#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:01:24 2019

@author: shubham
"""

import torch

x = torch.ones(2,2, requires_grad=True) #Tracks computation
print(x)

y = x + 2 # perfrom a tensor operation
print(y)


#y was created as a result of an operation, so it has a grad_fn
print(y.grad_fn)

#Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)
