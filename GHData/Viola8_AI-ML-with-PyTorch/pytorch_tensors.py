# Construct a 3x5 matrix, uninitialized (whatever values were in the allocated memory at the time will appear as the initial values)
import torch

a = torch.empty(3,5)
print(a)

# Construct a randomly initialized matrix.
b = torch.rand(4,4)
print(b)

# Construct a matrix filled zeros and of dtype long.
c = torch.zeros(2,2, dtype=torch.long)
print(c)
# ([[0,0],
#[0,0]])

# Construct a tensor directly from data.
d = torch.tensor([2,15,3.3])
print(d)                         # ([2.0000, 15.0000, 3.3000])

# Create a tensor based on an existing tensor.
# 1 creating tensor 3x3:
a = a.new_ones(3,3, dtype=torch.double)      # new_* methods take in sizes
print(a)
# 2 overriding dtype but with the same size:
a = torch.randn_like(a, dtype=torch.float)
print(a)
print(a.size())  # torch.Size([3,3]) torch.Size is in fact a tuple, so it supports all tuple operations.
