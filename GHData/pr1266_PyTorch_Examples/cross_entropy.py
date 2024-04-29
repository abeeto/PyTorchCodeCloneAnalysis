import torch
import torch.nn as nn
import numpy as np
import os

os.system('cls')

def cross_entropy_loss(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y = np.array([1, 0, 0])
# y_pred_good = np.array([0.7, 0.2, 0.1])
# y_pred_bad = np.array([0.1, 0.3, 0.6])
# l1 = cross_entropy_loss(y, y_pred_good)
# l2 = cross_entropy_loss(y, y_pred_bad)

# print(l1, l2)

loss = nn.CrossEntropyLoss()
y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 3*3
y_pred_good = torch.tensor([[.1, 1.0, 2.1], [2.0, 1.0, 0.1], [.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [.1, 1.0, 2.1], [.1, 3.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(l1, l2)

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(predictions1)
print(predictions2)