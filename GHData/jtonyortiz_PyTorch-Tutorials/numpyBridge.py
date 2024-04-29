from __future__ import print_function
import torch
import numpy as np

#NumPy Bridge:

#Converting a Torch Tensor to a NumPy Array
strng = "Converting a Torch Tensor to a NumPy Array"
print(strng)
a = torch.ones(5)
print(a)


b = a.numpy()
print(b)


#Add 1 to the numpy array:
strng = "Adding 1 to the array:"
print(strng)
a.add_(1)
print(a)
print(b)

#Converting NumPy Array to Torch Tensor
strng = "Converting NumPy Array to Torch Tensor"
print(strng)
a  = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


