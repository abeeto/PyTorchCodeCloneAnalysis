import torch

x= torch.randn(3, requires_grad=True)
print(x)
y=x+2
print(y)
z=y*y*2

print(z)

v=torch.tensor([0.1,1.0,.001])
z.backward(v)#dz/dx
print(x.grad)
