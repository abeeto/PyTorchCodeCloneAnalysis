import torch
import numpy as np

'''
1. Reshape
2. Squeeze() or Unsqueeze()
    - Removes or adds axes with length 1
    - To expand or shrink the rank
3. Flatten()
    - similar to t.reshape(-1)
    - t.flatten(start_dim, end_dim=end)
'''

# t.size() == t.shape
# rank = len(t.shape)

rand = torch.rand(3,6,2,1,5,1,10) #random matrix
print(rand.squeeze().shape)
print(rand.squeeze(3).shape)

img_batch = torch.rand(5, 3, 120, 60)
print(img_batch.flatten(1).shape)
print(img_batch.flatten(1,2).shape)

def flatten(t):
    return t.reshape(-1)
print(flatten(rand).shape)

# Number of elements
print("Number of Elements: ", img_batch.numel())
print("Another way: ", torch.tensor(img_batch.shape).prod())

t1 = torch.rand(5,5)
t2 = torch.rand(5,5)
t3 = torch.rand(5,5)
stacked = torch.stack((t1,t2,t3))
print(stacked.shape)
batch = stacked.unsqueeze(1)
print(batch.shape)

flattened = batch.flatten(1)
print(flattened.shape)