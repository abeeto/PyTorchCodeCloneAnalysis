# -*- coding: utf-8 -*-

import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.optim as optim

def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    #SRC: https://discuss.pytorch.org/t/fastest-way-of-converting-a-real-number-to-a-one-hot-vector-representing-a-bin/21578/2
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size(0), n_dims).scatter_(1, indexes, 1)
    one_hots = one_hots.view(*indexes.shape[:-1], -1)
    return one_hots

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST('mnist', train=False, download = True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = len(testset), shuffle=True, num_workers=0)

conv = torch.nn.Sequential(
    nn.Conv2d(1, 20, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2)
)

fullc = torch.nn.Sequential(
    nn.Linear(4*4*50, 500),
    nn.ReLU(),
    nn.Linear(500, 10))

conv.load_state_dict(torch.load('pytorchtestnet-conv.pt'))
fullc.load_state_dict(torch.load('pytorchtestnet-fullc.pt'))

loss_fn = torch.nn.CrossEntropyLoss()

parameterlist = list(conv.parameters()) + list(fullc.parameters())

x, y = next(iter(testloader))

y_pred = conv(x)
y_pred = fullc(y_pred.view(-1, 4 * 4 * 50))
y_pred = torch.argmax(y_pred, 1)

correct = ( y_pred == y )
correct = torch.sum(correct.int())
correct = 1 - correct.item() / y_pred.size(0)

print(round(correct * 100 * 100)/100)


