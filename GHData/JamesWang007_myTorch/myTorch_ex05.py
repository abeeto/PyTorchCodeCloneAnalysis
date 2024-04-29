# -*- coding: utf-8 -*-
# tensor attributes


import torch

torch.rand(5,3)
torch.randn(3,5)

# Via a string:
torch.device('cuda:0')
torch.device('cpu')
torch.device('cuda')

# Via a string and device ordinal:
torch.device('cuda', 0)
torch.device('cpu', 0)

torch.cuda.current_device() # current cuda device

# assign a gpu
cuda0 = torch.device('cuda:0')
torch.randn((2,3), device = cuda0)

