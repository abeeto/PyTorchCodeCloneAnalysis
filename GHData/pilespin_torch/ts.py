# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.optim as optim
from pathlib import Path

from Dataset import *

d = Dataset(784, 10)

# d.add(d.imageToArray("minimnist/0.png", "L"), d.selectedOutputToArray(0))
# d.add(d.imageToArray("minimnist/1.png", "L"), d.selectedOutputToArray(1))
# d.add(d.imageToArray("minimnist/2.png", "L"), d.selectedOutputToArray(2))
# d.add(d.imageToArray("minimnist/3.png", "L"), d.selectedOutputToArray(3))
# d.add(d.imageToArray("minimnist/4.png", "L"), d.selectedOutputToArray(4))
# d.add(d.imageToArray("minimnist/5.png", "L"), d.selectedOutputToArray(5))
# d.add(d.imageToArray("minimnist/6.png", "L"), d.selectedOutputToArray(6))
# d.add(d.imageToArray("minimnist/7.png", "L"), d.selectedOutputToArray(7))
# d.add(d.imageToArray("minimnist/8.png", "L"), d.selectedOutputToArray(8))
# d.add(d.imageToArray("minimnist/9.png", "L"), d.selectedOutputToArray(9))

d.addFolderWithLabel("dataset/mnist_png/training/", toTest=False)
d.addFolderWithLabel("dataset/mnist_png/testing/", toTest=True)

# d.printInput()
# d.printOutput()

# N       = 64    # batch size
D_in    = 784
H       = 15
D_out   = 10

# random.shuffle(d.getInput())

x = Variable(torch.Tensor(d.getInput()))
y = Variable(torch.Tensor(d.getOutput()), requires_grad=False)

x_test = Variable(torch.Tensor(d.getInputTest()))
y_test = Variable(torch.Tensor(d.getOutputTest()), requires_grad=False)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
# x = Variable(torch.randn(N, D_in))
# # print x
# y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.

my_file = Path("model")
if my_file.is_file():
    model = torch.load("model")
else:
    model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),   torch.nn.Tanh(),
            torch.nn.Linear(H, H),      torch.nn.Tanh(),
            torch.nn.Linear(H, D_out),  torch.nn.ReLU(),
        )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_test)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_test)
    print(t, loss.data[0])

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    if (t % 100 == 0):
        torch.save(model, "model")

print model