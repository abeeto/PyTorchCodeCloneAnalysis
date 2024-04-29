# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:43:20 2018

@author: trisgelar
"""

import torch

tensor = torch.Tensor([[3,4], [7,5]])
tensor

tensor.requires_grad_()

print(tensor.grad)

print(tensor.grad_fn)

out = tensor * tensor

out.requires_grad

print(out.grad)

print(out.grad_fn)

out = (tensor * tensor).mean()
print(out.grad_fn)
print(tensor.grad)

out.backward()
print(tensor.grad)
print(tensor.grad_fn)

new_tensor = tensor * tensor
print(new_tensor.requires_grad)

with torch.no_grad():
    new_tensor = tensor * tensor
    print("new_tensor = ", new_tensor)
    print("requires_grad for tensor = ", tensor.requires_grad)
    print("requires_grad for new_tensor = ", new_tensor.requires_grad)

    