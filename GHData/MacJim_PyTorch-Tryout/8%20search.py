import numpy as np
import torch


# Search within tensor.
x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
x = x.view(2, -1)

print(x)
print(x.shape)    # 2, 4

positions1 = np.transpose(np.argwhere(x == 5))
print(positions1)    # tensor([[1, 0]])
print(positions1.shape)    # 1, 2

positions2 = np.transpose(np.argwhere(x == 9))
print(positions2)    # tensor([], size=(0, 2), dtype=torch.int64)
print(positions2.shape)    # 0, 2 Wow it still has a size!
