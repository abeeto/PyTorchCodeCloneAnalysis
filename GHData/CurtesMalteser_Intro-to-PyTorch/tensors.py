import numpy as np
import torch

import helper

x = torch.rand(3, 2)
print(x)

y = torch.ones(x.size())
print(y)

z = x + y
print(z)

# get first row
print(z[0])

# with slices and print all rows for second column
print(z[:, 1:])

# Return a new tensor z + 1
z.add(1)

# Reshape
# Add 1 and update z tensor in-place
z.add_(1)

# Inspect tensor size
print(z.size())
print(torch.Size([3, 2]))

# chang from 3 by to 2 to 2 by 3
print(z.resize_(2, 3))

# Numpy to Torch and back
# Create Numpy array
a = np.random.rand(4, 3)
print(a)

# Create tensor from Numpy array
b = torch.from_numpy(a)
print(b)

# Convert back tensor to Numpy array
print(b.numpy())

# tensor and Numpy array share memory, so multipy tensor in-place will change the array too
c = b.mul_(2)
print(c)
print(c.numpy())
