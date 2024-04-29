import torch
import numpy as np

data = [-1,-2,1,2]
tensor = torch.FloatTensor(data) 

print(
    '\nnumpy', np.abs(data),
    '\ntorch', torch.abs(tensor)
) 