from __future__ import print_function
import numpy as np
import torch

a = torch.ones(5)
#print(a)

b = a.numpy()
#print(b)

a.add_(1)
print(a)
print(b)

#Returned tensor from torch.from_numpy shares memory with original numpy object
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

