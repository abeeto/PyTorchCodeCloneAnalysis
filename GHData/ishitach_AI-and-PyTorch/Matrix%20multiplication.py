Matrix multiplication using torch

import torch
torch.manual_seed(7)
features = torch.randn((1,5))
weigths = torch.randn_like(features)
bias = torch.rand((1,1))
sum = torch.sum(features)

sum1 = features.sum()

sum2 = weigths.view(5, 1)

print(torch.mm(features, sum2))
