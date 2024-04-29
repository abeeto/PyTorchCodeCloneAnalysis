
"""
Cross entropy example
"""
import numpy as np

Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
Y_pred3 = np.array([0.1, 0.1, 0.1])

print("loss1 = ", np.sum(-Y * np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y * np.log(Y_pred2)))
print("loss3 = ", np.sum(-Y * np.log(Y_pred3)))


"""
Softmax + CrossEntropy (logSoftmax + NLLLoss)
"""
import torch

loss = torch.nn.CrossEntropyLoss()

# Input is class, not one-hot
Y = torch.LongTensor([0])

Y_pred1 = torch.Tensor([[2.0, 1.0, 0.1]])
Y_pred2 = torch.Tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("PyTorch Loss1 = ", l1.item())
print("PyTorch Loss2 = ", l2.item())