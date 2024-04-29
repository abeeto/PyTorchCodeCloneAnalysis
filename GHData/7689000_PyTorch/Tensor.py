import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from torch import FloatTensor

x = Variable(torch.Tensor(4,4).uniform_(-4,5))
y = Variable(torch.Tensor(4,4).uniform_(-3,2))
# matrix multiplication
z = torch.mm(x,y)
print(z.size())
#print(’Requires Gradient : %s ’ % (z.requires_grad))
#print(’Gradient: %s ’ % (z.grad))
print(z.data)

# tensors’ definition
mat1 = torch.FloatTensor(4,4).uniform_(0,1)
print(mat1)
mat2 = torch.FloatTensor(5,4).uniform_(0,1)
print(mat2)
vec1 = torch.FloatTensor(4).uniform_(0,1)
print(vec1)
# scalar addition
print(mat1 + 10.5)
# scalar subtraction
print(mat2 - 0.20)
# vector and matrix addition
print(mat1 + vec1)
print(mat2 + vec1)
# matrix product
print(mat1 * mat1)