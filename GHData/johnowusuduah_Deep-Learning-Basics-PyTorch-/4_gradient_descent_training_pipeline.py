import numpy as np

# Order of Tutorial
# a. First Without PyTorch
# 1. Prediction: Manual
# 2. Gradients Computation: Manual
# 3. Loss Computation: Manual
# 4. Parameter Updates: Manual

# b. With PyTorch
# 1. Prediction: PyTorch Model
# 2. Gradients Computation: Autograd
# 3. Loss Computation: PyTorch Loss
# 4. Parameter Updates: PyTorch Optimizer

# i.*******************************************************
### MANUAL IMPLEMENTATION OF SIMPLIFIED LINEAR REGRESSION
# f = x * w
# w = 2
# f = x * 2

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return x * w


# loss = MSE
def loss(y, y_predicted):
    return np.mean(((y_predicted - y) ** 2))


# gradient
# MSE OR J = 1/N * (x*w - y)**2
# dJ/dw = 1/N * 2x * (x*w - y)
def gradient(x, y, y_predicted):
    return np.mean(np.dot(2 * x, y_predicted - y))


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):

    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dW = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dW

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: W = {w:.3f}, loss={l:0.8f}")


print(f"Prediction after training: f(5) = {forward(5):.3f}")

# ii.*******************************************************************************
### PYTORCH IMPLEMENTATION OF SIMPLIFIED LINEAR REGRESSION - WITH AUTOGRAD ONLY
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# since we are interested in the gradient of the loss function with respect to w, we
# need to specify requires_grad
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# forward pass function and loss function remains the same
# computation of gradient function replace by autograd
def forward(x):
    return x * w


def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):

    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients computation = backward_pass
    l.backward()

    # update weights - this operation should not be part of the gradient computational graph
    # so we wrap it in a with statement
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients - whenever we call l.backward(), it will accumulate the gradients
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: W = {w:.3f}, loss={l:0.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")


# iii.**************************************************************************************
### PYTORCH IMPLEMENTATION OF SIMPLIFIED LINEAR REGRESSION - WITH LOSS AND OPTIMIZER CLASSES
# NOTES
# General Training Pipeline in PyTorch
# 1. Design Model(input size, output size, forward pass)
# 2. Construct Loss and Optimizer
# 3. Training Loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# since we are interested in the gradient of the loss function with respect to w, we
# need to specify requires_grad
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# only forward pass function remains the same
# computation of gradient function replace by autograd
def forward(x):
    return x * w


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 100

# loss - use loss provided by PyTorch
loss = nn.MSELoss()
# optimizer
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(1, n_iters):

    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients computation = backward_pass
    l.backward()

    # update weights - optimizer
    optimizer.step()

    # zero gradients - whenever we call l.backward(), it will accumulate the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch {epoch}: W = {w:.3f}, loss={l:0.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")


# iv.*******************************************************************************************************************
## PYTORCH IMPLEMENTATION OF SIMPLIFIED LINEAR REGRESSION - WITH PREDICTION(OR FORWARD PASS), LOSS AND OPTIMIZER CLASSES
# we will not need to define weights any more
# NOTES
# General Training Pipeline in PyTorch
# 1. Design Model(input size, output size, forward pass)
# 2. Construct Loss and Optimizer
# 3. Training Loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import torch.nn as nn

# features and labels should be multi-dimensional
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# multidimensional feature matrix
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

# we do not define the weights any longer
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# prediction (forward pass)
# nn.Linear requires X and y to be multi-dimensional arrays
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

# Training
learning_rate = 0.01
n_iters = 100

# loss - use loss provided by PyTorch
loss = nn.MSELoss()
# optimizer - here we are no longer using weights so we specify model.parameters()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

for epoch in range(1, n_iters):

    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward_pass
    l.backward()

    # update weights - optimizer
    optimizer.step()

    # zero gradients - whenever we call l.backward(), it will accumulate the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        # we can unpack the weights
        [w, b] = model.parameters()
        print(f"epoch {epoch}: W = {w[0][0].item():.3f}, loss={l:0.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")


# v.*********************************************************************
### CUSTOM LINEAR REGRESSION MODEL
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        # i need to understand why we do this
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)


model_0 = LinearRegression(input_size, output_size)
