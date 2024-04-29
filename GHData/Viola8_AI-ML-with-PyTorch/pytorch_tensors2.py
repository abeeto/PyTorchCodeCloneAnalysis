# Conversion between NumPy ndarray and Tensor
# Both ndarray and Tensor share the same memory storage. Change value from either side will affect the other.
import numpy as np
import torch

a = np.array([1, 2, 3])
c = torch.from_numpy(a)         # Convert a numpy array to a Tensor
print(a)  # [1 2 3]
print(c) # tensor([1,2,3], dtype=torch.int32)

b = c.numpy()                   # Tensor to numpy
b[1] = -1                       # Numpy and Tensor share the same memory
assert(a[1] == b[1])            # Change Numpy will also change the Tensor
print(b) # [ 1 -1  3]

# Initialize Tensor with a range of value
d = torch.arange(5)             # similar to range(5) but creating a Tensor
print(d)   # tensor([0,1,2,3,4])
d = torch.arange(0, 10, step=2)  # Size 5. Similar to range(0, 5, 1)
print(d) # tensor([0,2,4,6,8])

d = torch.arange(9)
print(d.view(3, 3))

# Initialize a linear or log scale Tensor
g = torch.linspace(1, 10, steps=10) # Create a Tensor with 10 linear points for (1, 10) inclusively
print(g)
g = torch.logspace(start=-10, end=10, steps=5) # Size 5: 1.0e-10 1.0e-05 1.0e+00, 1.0e+05, 1.0e+10
print(g)

# Initialize a ByteTensor
print(torch.ByteTensor([0, 1, 1, 0])) # tensor([0, 1, 1, 0], dtype=torch.uint8)
