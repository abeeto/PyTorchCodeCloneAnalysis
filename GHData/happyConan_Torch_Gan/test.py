import torch
import numpy as np
import torch.nn as nn
import math
import torch.tensor as tensor

label = torch.full((6,), 1)
label.fill_(-1)
l=label.mean().item()
print(l)
print(torch.cuda.current_device())