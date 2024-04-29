import torch
from __future__ import print_function
print("Torch version: ",torch.__version__)
print(torch.rand(5,4))
torch.cuda.is_available()
