## Basic gradient descent in PyTorch.

import torch
from torch.autograd import Variable
import numpy as np
import time


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2).sum()))


def forward(x, e):
    """Forward pass for our fuction"""
    return x.pow(e.repeat(x.size(0)))



# Let's define some settings
n = 1000  # number of examples
learning_rate = 5e-10
target_exp = 2.0  # real value of the exponent will will try to find

# Model definition
x = Variable(torch.rand(n) * 10, requires_grad=False)

# Model parameter and it's true value
exp = Variable(torch.FloatTensor([target_exp]), requires_grad=False)
exp_hat = Variable(torch.FloatTensor([4]), requires_grad=True)  # just some starting value, could be random as well
y = forward(x, exp)

# a couple of buffers to hold parameter and loss history
loss_history = []
exp_history = []

# Training loop
loss = 10000000000000000000
while loss > 0.0001:
    # print("Iteration %d" % i)

    # Compute current estimate
    y_hat = forward(x, exp_hat)

    # Calculate loss function
    loss = rmse(y, y_hat)

    # Do some recordings for plots
    loss_history.append(loss.data[0])
    exp_history.append(y_hat.data[0])

    # Compute gradients
    loss.backward()

    print("loss = %s" % loss.data[0])
    print("exp = %s" % exp_hat.data[0])

    # Update model parameters
    exp_hat.data -= learning_rate * exp_hat.grad.data
    exp_hat.grad.data.zero_()
