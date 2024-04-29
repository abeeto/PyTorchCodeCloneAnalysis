#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:10:45 2019

@author: matt
"""

import torch
import numpy as np

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)

print(x)


a = np.array([[1, 2, 3],[4,5,6]])

b = torch.tensor(a)


print(torch.cuda.is_available())

dev = torch.device("cuda")

b
b = b.to(dev)
print(b)
print(b.to("cpu"))


x = torch.ones(2, 2, requires_grad=True)

print(x)
y = x + 2

print(y)

z = y * y * 3
out = z.mean()
print(z, out)



#a = torch.randn(2, 2)
#a = ((a*3)/(a-1))
#print(a.requires_grad)
#a.requires_grad=True
#a
#b = (a*a).sum()
#print(b.grad_fn)

out.backward()
print(x.grad)

#%%
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
