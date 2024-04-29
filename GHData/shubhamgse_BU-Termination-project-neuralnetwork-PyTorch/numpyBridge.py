#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:47:25 2019

@author: shubham
"""

import torch
import numpy as np

#Converting torch tensor to numpy

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)




#Converting numpy to torch tensor

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1, out=a)
print(a)
print(b)
