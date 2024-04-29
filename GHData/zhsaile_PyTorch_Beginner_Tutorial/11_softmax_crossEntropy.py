import numpy as np
import torch
import torch.nn as nn

print('# softmax with numpy')

y = np.array([1., 2., 3.], dtype=np.float32)

output = np.exp(y) / np.sum(np.exp(y))

print(output)
print(np.sum(output))

print('\n# cross-entropy with numpy')

def softmax(y):
    return np.exp(y) / np.sum(np.exp(y))


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0], dtype=np.float32)
Y_good = np.array([.7, .2, .1])
Y_bad = np.array([.1, .3, .6])
l1 = cross_entropy(Y, softmax(Y_good))
l2 = cross_entropy(Y, softmax(Y_bad))

print(l1)
print(l2)

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples x nclasses = 1 x 3
Y_good = torch.tensor([[0.7, 0.2, 0.1]])
Y_bad = torch.tensor([[0.1, 0.3, 0.6]])

l1 = loss(Y_good, Y)
l2 = loss(Y_bad, Y)
print(l1.item())
print(l2.item())

#_, predictions1 = torch.max(Y_good, 1)
#_, predictions2 = torch.max(Y_bad, 1)

#print(predictions1)
#print(predictions2)
