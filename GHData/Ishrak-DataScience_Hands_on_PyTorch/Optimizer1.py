from pickletools import optimize
import torch
weights=torch.ones(4,requires_grad=True)
optimizer=torch.optim.SGD(weights,lr=0.01)
optimizer.step()
optimizer.zero_grad()