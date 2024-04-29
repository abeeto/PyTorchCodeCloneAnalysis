import torch
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package

# here's a one dimensional array the pytorch way (allowing GPU computionals)

x1 = torch.Tensor([1, 2, 3, 4])

# here's a two dimensional array (i.e. of size 2 x 4):
x2 = torch.Tensor([[5, 6, 7, 8], [9, 10, 11, 12]])

# here's a three dimensional array (i.e. of size 2 x 2 x 4):
x3 = torch.Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])

# x1

print("----------------------------------------")
print(x1[0])
print("----------------------------------------")

# x2

print("----------------------------------------")
print(x2[0, 0])
# prints 5.0; the first entry of the first vector

print("----------------------------------------")
print(x2[0, :])
# prints 5, 6, 7, 8; all the entries of the first vector

print("----------------------------------------")
print(x2[:, 2])
print("----------------------------------------")
# prints 7, 11; all the third entries of each vector vector

x1_node = Variable(x1, requires_grad=True)
# we put our tensor in a Variable so we can use it for training and other stuff later

print("----------------------------------------")
print(x1_node)
print("----------------------------------------")