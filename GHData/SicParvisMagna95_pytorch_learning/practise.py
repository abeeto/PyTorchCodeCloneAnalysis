import torch
import torch.nn as nn
from torch.optim import Adam

model = nn.Sequential(nn.Conv2d(1,20,5))

optimizer = Adam(model.parameters())
g = optimizer.param_groups
print(optimizer.param_groups)
