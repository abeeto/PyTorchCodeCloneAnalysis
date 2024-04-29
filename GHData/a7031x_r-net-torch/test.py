import utils
import torch
import torch.nn as nn

a = [
    [1,2,3,4],
    [-2,-4,-6,-8]
]
a = torch.tensor(a).float()
norm = nn.LayerNorm(4)
b = norm(a)

print(b.tolist())