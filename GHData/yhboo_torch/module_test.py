import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np




x_np = np.asarray([[1,2],[3,4]], dtype = 'float32')
x_tc = torch.from_numpy(x_np)
y_tc = torch.from_numpy(x_np)

print(x_np)
print(x_tc)

z_tc = torch.mm(x_tc, y_tc)

print(z_tc)
nn.Conv2d