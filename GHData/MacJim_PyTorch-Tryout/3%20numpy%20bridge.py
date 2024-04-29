# Source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np


# MARK: Numpy array from torch tensor.
a = torch.ones(5)
print(type(a))    # torch.Tensor
print(a)

b = a.numpy()
print(type(b))    # numpy.ndarray
print(b)

# Values of CPU tensors and numpy arrays share their memory addresses.
a += 2
print(a)    # 3 3 3 3 3
print(b)    # 3 3 3 3 3


# MARK: Torch tensor from numpy array.
a = np.ones((3, 3))
b = torch.from_numpy(a)
print(a)
print(b)

# Values of CPU tensors and numpy arrays share their memory addresses.
a += 2
print(a)
print(b)
