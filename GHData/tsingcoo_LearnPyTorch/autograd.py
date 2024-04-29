# The autograd package provides automatic differentiation for all operations on Tensors.

from __future__ import print_function
from torch.autograd import Variable
import torch

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
print(z)

out = z.mean()

out.backward()

print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2

while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
