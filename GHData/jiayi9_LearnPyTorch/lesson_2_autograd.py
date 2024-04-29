# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:22:35 2020

@author: LUI8WX
"""
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
# https://pytorch.org/docs/stable/autograd.html#function

import torch

# x -> y -> z -> out

x = torch.ones(2, 2, requires_grad=True)

print(x.requires_grad)

y = x + 2
y

print(y.requires_grad)

print(y.grad_fn)

z = y * y * 3

print(z.requires_grad)

out = z.mean()

print(x)
print(y)
print(z)
print(out)

#tensor([[1., 1.],
#        [1., 1.]], requires_grad=True)     
#tensor([[3., 3.],
#        [3., 3.]], grad_fn=<AddBackward0>)   y = x + 2
#tensor([[27., 27.],
#        [27., 27.]], grad_fn=<MulBackward0>)   z = y * y * 3
#tensor(27., grad_fn=<MeanBackward1>)         out = z.mean()


out.backward()

#gradients of out with respect to x vector
x.grad



# intermediate result has no grad
y.grad
z.grad



##########   multiply element by element #########
x = torch.tensor([5, 3, 4])
y = torch.tensor([5, 3, 9])
x*y





#############################################
####    Cookup example #########
import torch
x = torch.tensor([3], requires_grad=True, dtype=torch.float)
y = x*x
y.backward()
x.grad

x = torch.tensor([3,4], requires_grad=True, dtype=torch.float)
y = x*x
y.backward()
x.grad
# RuntimeError: grad can be implicitly created only for scalar outputs

x = torch.tensor([3,4], requires_grad=True, dtype=torch.float)
y = x*x
z1 = y.std()
z1.backward()
x.grad


x = torch.tensor([3,4], requires_grad=True, dtype=torch.float)
y = x*x*x
z2 = y.mean()
z2.backward()
x.grad

#(3*x^2)/2
#3*3**2/2





###########  Set requires_grad_ #############

#a = torch.randn(2, 2)
#
#a = ((a * 3) / (a - 1))
#
#print(a.requires_grad)
#a.requires_grad_(True)
#print(a.requires_grad)
#
#b = (a * a).sum()
#
#print(b.grad_fn)





#########   element by element grad  ##########  

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

x.grad


# my example 
# element by element gradient
x = torch.tensor([3,4], requires_grad=True, dtype=torch.float)
y = x*x
v = torch.tensor([2,3], dtype = torch.float)
y.backward(v)
x.grad

x = torch.tensor([3,4], requires_grad=True, dtype=torch.float)
y = x*x
v = torch.tensor([1,1], dtype = torch.float)
y.backward(v)
x.grad


2*3*2
2*4*3





###########  Specify NO grad!  #############

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
    




########## detach ##############
    
x = torch.ones(2, 2, requires_grad=True)

print(x.requires_grad)

y = x + 2
y
    
print(x.requires_grad)

x2 = x.detach()

print(x2.requires_grad)

# all elements are equal (x and x2)
print(x.eq(x2).all())
