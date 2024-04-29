# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:47:30 2020

@author: rrite
"""
import torch
import numpy as np
a = np.random.randn(4,3)
# Converting numpy arrays to torch tensors
b = torch.from_numpy(a)
print(a)
print(b)
# Converting torch tensors to numpy arrays
c = torch.randn((4,3))
d = c.numpy()
print(c)
print(d)

""" Memory is shared between the Numpy array and Torch tensor, 
 so if u change the values in-place of one object,
the other object will change as well """

