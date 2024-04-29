#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:07:06 2020

reference: Understanding PyTorch with an example: a step by step tutorial
towardsdatascience

@author: pengfei
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
#Data generation

np.random.seed(42)
x = np.random.rand(100,1)
y = 1 + 2 * x + 0.1 * np.random.randn(100,1)

# Shuffles the indices

idx = np.arange(100)
np.random.shuffle(idx)

# use first 80 random indices for train

train_idx = idx[:80]
# use the remaining indices for validition
val_idx = idx[80:]

# Generates train and validation sets

x_train, y_train = x[train_idx],y[train_idx]
x_val, y_val = x[val_idx],y[val_idx]

#%% Plot data
plt.plot(x,y,"*")
plt.plot(x_train,y_train,"^")
plt.plot(x_val, y_val,".")
plt.show()

#%% Initializes parameters "a" and "b" randomly. parameters or weight
np.random.seed(42)
a = np.random.rand(1)
b = np.random.rand(1)

print(a,b)

# the following is hyperparameters
lr = 1e-1
n_epochs = 1000

for epoch in range(n_epochs):
    #Computes our model's predicted output
    yhat = a + b * x_train
    # how wring is our model, that is the error
    error = (y_train - yhat)
    # It is a regression, so it computes mean square error MSE
    loss = (error ** 2).mean()
    #Computes gradient for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    #Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a,b)
    
# Sanity Check: do we get the same results as our gradient descent
from sklearn.linear_model import LinearRegression

linr = LinearRegression()
linr.fit(x_train, y_train)
print("a and b from sklearn fitting is: {} {}".format(linr.intercept_,linr.coef_[0]))

#%%
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot

#%%

device = "cuda" if torch.cuda.is_available() else "cpu"

# our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful

print(type(x_train),type(x_train_tensor),x_train_tensor.type())

#%% we specify the device at the moment of creation - Recommended.

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)
b = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    loss.backward()
    print(a.grad)
    print(b.grad)
    
    with torch.no_grad():
        a-= lr * a.grad
        b-= lr * b.grad
    
    a.grad.zero_()
    b.grad.zero_()

print(a,b)

make_dot(yhat)



#%% Pure torch linear regression
torch.manual_seed(42)
a = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)
b = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)
print("\nPure Torch calculation, initial parameters: a-{}, b-{}".format(a,b))
lr = 1e-1 
n_epochs = 1000

# Defines a SGD optimizer to update 
optimizer = optim.SGD([a,b],lr = lr)

for epoch in range(n_epochs):
    
    yhat = a + b * x_train_tensor # predictions
    # loss function creation
    error = y_train_tensor - yhat # labels and predictions difference
    # loss function
    loss = (error ** 2).mean()
    
    #1. calculating the gradients.
    loss.backward()
    # no more manual update our parameters, the coefficients you want to calculate
    # with torch.no_grad():
    #   a -= lr * a.grad
    #   b -= lr * b.grad
    # Update our parameters.
    optimizer.step()    
    
    # No more telling PyTorch to let gradients go!
    # a.grad.zero_()
    # b.grad.zero_()
    # Empty the grads.
    optimizer.zero_grad()
    
print("\nPure Torch calculated results: a-{}, b-{}".format(a,b))
    
#%% Pure torch linear regression, loss function
torch.manual_seed(42)
a = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)
b = torch.randn(1, requires_grad = True, dtype = torch.float, device = device)
print("\nPure Torch calculation, initial parameters: a-{}, b-{}".format(a,b))
lr = 1e-1 
n_epochs = 1000

#Defines loss function
loss_fn = nn.MSELoss(reduction = "mean")
# Defines a SGD optimizer to update 
optimizer = optim.SGD([a,b],lr = lr)

for epoch in range(n_epochs):
    
    yhat = a + b * x_train_tensor # predictions
    # loss function creation
    #loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)
    
    #1. calculating the gradients.
    loss.backward()
    # no more manual update our parameters, the coefficients you want to calculate
    # with torch.no_grad():
    #   a -= lr * a.grad
    #   b -= lr * b.grad
    # Update our parameters.
    optimizer.step()# update paramters, there paramters are moving to true value, using lr. 
    
    # No more telling PyTorch to let gradients go!
    # a.grad.zero_()
    # b.grad.zero_()
    # Empty the grads.
    optimizer.zero_grad()
    
print("\nPure Torch calculated results: a-{}, b-{}".format(a,b))

#%% Create class for model for predictions
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # to make "a" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
    
    def forward(self,x):
        # computes the outputs /predictions
        return self.a + self.b * x # this is the real model for predictions

model = ManualLinearRegression().to(device)
print(model.state_dict())

lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(),lr = lr)

for epoch in range(n_epochs):
    model.train()
    
    # no more manual predictions!
    # yhat = a + b * x_tensor
    yhat = model(x_train_tensor)
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())








