import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1])  # n_samples * n_classes = 3 * 3

Y_pred_good = torch.tensor([[1.0, 0.1, 2.0], [3.0, 0.5, 0.3], [0.2, 4.0, 0.1]])
Y_pred_bad = torch.tensor([[2.0, 0.1, 1.0], [0.0, 0.5, 3.3], [0.2, 0.0, 4.1]])

# Cross entropy loss
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1)
print(l2)

# Choses the highest val
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)
