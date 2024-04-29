import torch
import numpy as np 

t1 = torch.tensor([
    [1,2],
    [3,4]
], dtype=torch.float32)

t2 = torch.tensor([
    [9,8],
    [7,6]
], dtype=torch.float32)

#first axis
print(t1[0])

#second axis
print(t1[0][0])

#corresponding elements
print(t1[0][0])
print(t2[0][0])

#addition
t3 = t1 + t2
print(t3)
