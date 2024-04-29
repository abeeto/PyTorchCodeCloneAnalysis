import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

x_data = [1,2,3,4]
y_data = [2,4,6,8]

#Init weight
w = Variable(torch.Tensor([1.0]), requires_grad=True)

# Build a linear layer.
def forward(x):
    return x * w

# Loss function


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

for epoch in range(500):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        # Backpropagation.
        l.backward()

        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  13, forward(13).data[0])