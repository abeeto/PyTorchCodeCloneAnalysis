import torch
import torch.nn as nn 
import numpy as np 

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y must be one-hot-encoded
Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

#in pytorch

loss = nn.CrossEntropyLoss()
# 3 samples
Y = torch.tensor([2,0,1])
# nsamples x nclasses = 3x3
Y_pred_good = torch.tensor([[0.1,1.0,2.0],[2.0,1.0,0.1],[0.1,3.0,0.1]]) #raw values

Y_pred_bad = torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]]) #raw values

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 numpy: {l1.item()}')
print(f'Loss2 numpy: {l2.item()}')

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)

print(prediction1)
print(prediction2)
