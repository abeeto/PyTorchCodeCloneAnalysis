#########################################################################
# In this script, we will deal with several problems associated with
# optimization and see how momentum can improve results.
#########################################################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)


# Plot the cubic
def plot_cubic(w, optimizer):
    LOSS = []
    # parameter values
    W = torch.arange(-4, 4, 0.1)
    # plot the loss function, with the linear.weight changing from -4 to 3.9
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(cubic(w(torch.tensor([[1.0]]))).item())
    # print(w.state_dict()['linear.weight'][0])
    w.state_dict()['linear.weight'][0] = 4.0
    n_epochs = 10
    parameter = []
    loss_list = []

    # n_epochs
    # Use PyTorch custom module to implement a polynomial function
    for n in range(n_epochs):
        optimizer.zero_grad()
        # loss: the cubic of the linear.weight
        loss = cubic(w(torch.tensor([[1.0]])))
        loss_list.append(loss)
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        # SGD: \theta = \theta - lr * derivative of (\theta * 1.0)^3 w.r.t. \theta
        loss.backward()
        optimizer.step()
    plt.figure()
    plt.plot(parameter, loss_list, 'ro', label='parameter values')
    plt.plot(W.numpy(), LOSS, label='objective function')
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.legend()
    plt.show()


# Plot the fourth order function and the parameter values
def plot_fourth_order(w, optimizer, std=0, color='r',
                      paramlabel='parameter values', objfun=True):
    W = torch.arange(-4, 6, 0.1)
    LOSS = []
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(fourth_order(w(torch.tensor([[1.0]]))).item())
    w.state_dict()['linear.weight'][0] = 6
    n_epochs = 100
    parameter = []
    loss_list = []

    # n_epochs
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = fourth_order(w(torch.tensor([[1.0]]))) + std * torch.randn(1, 1)
        loss_list.append(loss)
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        loss.backward()
        optimizer.step()

    # Plotting
    plt.figure()
    if objfun:
        plt.plot(W.numpy(), LOSS, label='objective function')
    plt.plot(parameter, loss_list, 'ro', label=paramlabel, color=color)
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.legend()
    plt.show()


# Create a linear model
class one_param(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        super(one_param, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Define a function to output a cubic
def cubic(yhat):
    out = yhat ** 3
    return out


# Create a function to calculate the fourth order polynomial
def fourth_order(yhat):
    out = torch.mean(2 * (yhat ** 4) - 9 * (yhat ** 3) -
                     21 * (yhat ** 2) + 88 * yhat + 48)
    return out


# Create a one parameter object
w = one_param(1, 1)
# print(w.state_dict()['linear.weight'][0])

############### Cubic #######################
# Create a optimizer without momentum
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0)
# Plot the model
plot_cubic(w, optimizer)

# Create a optimizer with momentum
optimizer_m = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0.9)
# Plot the model
plot_cubic(w, optimizer_m)

############### Forth order #######################
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0)
plot_fourth_order(w, optimizer)
plot_fourth_order(w, optimizer, std=10)
optimizer_m = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer_m)
plot_fourth_order(w, optimizer_m, std=10)
