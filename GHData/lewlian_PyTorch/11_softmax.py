import torch
import torch as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

# creating it in tensor

x = torch.tensor([2.0,1.0,0.1])
output = torch.softmax(x, dim=0) #computes it along the first axis)
print(outputs)