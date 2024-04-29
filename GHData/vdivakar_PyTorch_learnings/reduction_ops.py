import torch
import numpy as np

t = torch.rand(2,5)
print(t.sum())
print(t.mean())
print(t.std())

print(t.mean(dim=0))
print(t.mean(dim=1))

print(t[0])
print(t[1])
print((t[0]+t[1]) == t.sum(dim=0))

print('*'*20)
print((t[:,0]+t[:,1]+t[:,2]+t[:,3]+t[:,4]) == t.sum(dim=1))
print(t[:,0]+t[:,1]+t[:,2]+t[:,3]+t[:,4])
print(t.sum(dim=1))
print(t[:,0], t[:,1], t[:,2])

print('*'*20, t.shape, "argMax")
print(t.max(dim=1).indices)
print(t.argmax(dim=1))
print(t.max(dim=1))

print('*'*20, t.shape, "access")
print(t.max().item())
print(t.max())
