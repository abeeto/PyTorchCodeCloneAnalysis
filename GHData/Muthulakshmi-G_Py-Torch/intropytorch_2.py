import torch
import torch.autograd as grad

import torch.functional as F

x=torch.randn(2,2)

y=torch.randn(2,2)

print(x.requires_grad,y.requires_grad)

z=x+y

print(z.grad_fn)


x=x.requires_grad_()
y=y.requires_grad_()


z=x+y

print(z)
print(z.grad_fn)
print(z.requires_grad)

print("===========")

new_z=z.detach()

print(new_z)

print(new_z.grad_fn)



data=torch.randn(5)
print(data)

print(F.softmax(data,dim=0))
print(F.softmax(data,dim=0).sum())

print(F.softmax(data,dim=0).sum())

print(F.log_softmax(data,dim=0))
1
