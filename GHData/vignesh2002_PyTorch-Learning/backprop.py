"""
BACKPROPOGATION
"""

import torch

## x -> a(x) -> y -> b(y) -> z
# we need to minimize the output z so we take the derivative w.r.t. x
# we do this usig the chain rule
# dz/dx = dz/dy . dy/dx; where dz/dy and dy/dx are called local gradients
# need for local gradients - to compute loss function which needs to be minimized 
# e.g. d(Loss)/dx = d(Loss)/dz . dz/dx

## Three step process
# 1). Forward pass: Compute Loss
# 2). Compute local gradients
# 3). Backward pass: Compute d(Loss)/d(Weight) using chain rule 

## Example using Linear Regression: x = 1, y = 2, w = 1;
# 1). Forward Pass:
# x * w -> y_hat;    y_hat - y -> s;    s^2 -> loss
# y_hat = w.x;    loss = (y_hat - y)^2 = (wx - y)^2;
# 2). Local Gradients: 
# d(Loss)/ds = 2s;    ds/d(y_hat) = d(y_hat - y)/d(y_hat) = 1;    d(y_hat)/dw = d(wx)/dw = x;
# 3). Backward Pass:
# d(Loss)/dw = d(Loss)/d(y_hat) . d(y_hat)/dw = -2 . x = -2

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

##forward pass: gets and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

##backward pass
loss.backward()
print("gradient =", w.grad)

## update weights
## next forward and backward