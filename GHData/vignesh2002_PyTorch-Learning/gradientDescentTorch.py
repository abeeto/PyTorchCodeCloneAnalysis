"""
GRADIENT DESCENT
Using only Torch
"""

import torch

## f = w * X
# E.g. f = 2 * X
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

## model prediction
def forward(x):
    return w*x
## loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')  

## Training 
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    ## prediction = forward pass
    y_pred = forward(X)

    ## loss
    l = loss(Y, y_pred)

    ## gradients
    # dl/dw
    l.backward()

    ## update weights
    # to avoid gradient calc to be part of computational graph:
    with torch.no_grad():
        w -= learning_rate * w.grad

    ## zero gradients to sum grads in w.grad
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')