import torch
import torch.nn as nn
import math
import torch.nn.functional as F

input = torch.randn(3, 4)
b = F.softmax(input, dim=0)

print(b)

d = torch.max(b, dim=0)    # 按列取max,
print(d)
