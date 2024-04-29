# #SofMax_and_crossEntropy

# # S(yi) =  e**yi / sigma(e**yj)

# import torch
# import torch.nn as nn
# import numpy as np

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)


# x = np.array([2.0, 1.0, 0.1])

# outputs = softmax(x)
# print('softmax: ', outputs)


# x = torch.tensor([2.0, 1.0, 0.1])
# outputs = torch.softmax(x, dim=0)
# print(outputs)


# # one hot encoded class labels 

# def cross_entropy_loss(actual, pred):
#     loss = -np.sum(actual * np.log(pred))

#     return loss # / float(pred.shape[0])
# Y = np.array([1, 0, 0])
#  # y_preds has probabilites

# Y_preds_good = np.array([0.7, 0.2, 0.1])
# Y_preds_bad = np.array([0.1, 0.3, 0.6])
# l1 = cross_entropy_loss(Y, Y_preds_good)
# l2 = cross_entropy_loss(Y, Y_preds_bad)

# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')


import torch 
import torch.nn as nn
import numpy as np 

loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])
# n_sample x n_class = 3x3
Y_pred_good = torch.tensor([2.0, 1.0, 0.1])
Y_pred_bad = torch.tensor([0.5, 2.0, 0.3])

print(Y_pred_bad)
l1 = loss(Y_pred_good, Y)
l2  = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, pred = torch.max(Y_pred_good, 1)
_, pred2 = torch.max(Y_pred_bad, 1)

print(pred, pred2)