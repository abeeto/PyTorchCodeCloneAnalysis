import torch
import numpy as np

print(torch.__version__)

print("Tensors -- ")

x = torch.empty(5,5)
print("empty 2d\n"+str(x))

x = torch.rand(5,5)
print("rand 2d\n"+ str(x))

x = torch.zeros(5, 3, dtype=torch.long)
print("zeroes 2d\n"+str(x))

y = torch.zeros(5, 3, 5)
print("empty 3d\n"+str(y))

print("Additions -- ")

y = torch.rand(5, 3)
x = torch.rand(5, 3)
z = torch.zeros(500,2)
torch.add(y,x, out=z)
print("add 2d\n")
for i,j,k in zip(x,y,z):
	print("{} + {} = {}".format(i, j, k))
print(z)


x = torch.tensor([[1, 2], [3, 4], [5, 7]])
print("tensor with list\n"+str(x))

x = x.new_ones(5,3, dtype=torch.double)
print("new ones\n" + str(x))

x = torch.randn_like(x, dtype=torch.float)
print("randn_like\n" + str(x))

print(x.size())

a = np.ones(5)
b = torch.from_numpy(a)
print("numpy\n" + str(a) +"\n-> tensor\n"+ str(b))
