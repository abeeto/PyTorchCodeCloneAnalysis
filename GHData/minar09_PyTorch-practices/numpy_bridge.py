from __future__ import print_function
import torch
import numpy as np

a = torch.ones(5, 2)
print(a)

b = a.numpy()
print(b)

a.add_(2)
print(a)
print(b)

a = np.ones([5, 2])
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
np.add(a, 2, out=a)
print(a)
print(b)