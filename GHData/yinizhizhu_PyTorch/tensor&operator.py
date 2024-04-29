from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

print(x.size())

y = torch.rand(5, 3)            #addition: syntax 1
print(x + y)

print(torch.add(x, y))          #addition: syntax2


result = torch.Tensor(5, 3)     #addition: giving an output tensor
torch.add(x, y, out=result)
print(result)

#Any operation taht mutates a tensor in-place is post-fixed with and '_'.
#For example: x.copy_(y), x.t_(), will change x
y.add_(x)                       #addition: in-place
print(y)

#We can use standard numpy-like indexing with all bells and whistles!
print(y[:, 1])