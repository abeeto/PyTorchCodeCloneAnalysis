'''
------------------------------------
AUTOMATIC DIFFERENTIATION ON PYTORCH
------------------------------------
'''

import torch
import numpy as np

# set requires_grad as True to include it in the acyclic graph
# use .backward() for backpropogation from this tensor
x=torch.ones(2,2,requires_grad=True)
print x

y=x+2
print y

# since y was created by an operation, it has a gradients_function
print y.grad_fn

z=y*y*3
out=z.mean()
print z,'\n',out

# by default, requires_grad is False
# can change it later using .requires_grad_(True)

# Starting backprop

# scalar case
out.backward()
print x.grad # d(out)/dx

# non-scalar case
x=torch.randn(3,requires_grad=True)
y=x*2
while y.data.norm()<1000 : 
	y=y*2
print y
v=torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v) # like learning rates
print x.grad

