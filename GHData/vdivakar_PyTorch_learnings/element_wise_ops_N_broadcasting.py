import torch
import numpy as np

t1 = torch.rand(2,2)
t2 = torch.rand(2,2)

print(t1)
print(t2)
print(t1+t2)

print(t1 + 100)

print(np.broadcast_to(100, (5,5)))

t1 = torch.tensor([[2,3], [4,5]]) ## (2,2)
t2 = torch.tensor([100, 100])     ## (2)
t3 = torch.tensor([[100, 100]])   ## (1,2)

print(t1+t2) ## t2 will be broadcasted to (2,2)
print(t1+t3) ## --''---
print(t1 + torch.tensor(np.broadcast_to(t2, t1.shape)))
print(t1 + torch.tensor(np.broadcast_to(t3, t1.shape)))

'''Bool operations'''
print(t1 < 102.5)