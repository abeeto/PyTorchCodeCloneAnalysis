import torch
import numpy as np


## Generatre random numbers and basic operations
x = torch.empty(2,2)
print(x)

x = torch.zeros(2,2)
print(x)

x = torch.ones(2,2)
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
z = x+y
print(z)

z = torch.add(x,y)
print(z)

y.add_(x)
print(y)

x = torch.rand(2,2)
y = torch.rand(2,2)
z = torch.mul(x,y)
print(z)


z = torch.div(x,y)
print(z)

x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[0,0].item())

## Reshape

x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)

z = x.view(-1,8)
print(z)

## Convert to NumPy

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))

## On cuda

if(torch.cuda.is_available()):
    print("Yes")
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    print(z)
    z = z.to('cpu')
    print(z.numpy())

    