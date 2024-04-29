from __future__ import print_function
import numpy as np
import torch

x = torch.zeros(5, 3, dtype=torch.long)
y = torch.ones(5, 3, dtype=torch.long)

a = np.ones(5)
b = torch.from_numpy(a)

print(b)
