# Creating a model that predicts crop yields for apples and oranges (target variables)
# by looking at the average temperature, rainfall, and humidity (input variables or features)

import torch
import numpy as np

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

#model
def model(x):
    return x @ w.t() + b

#mse
def mse(t1, t2):
    diff = t1-t2
    return torch.sum(diff*diff) / diff.numel()

#Adjust weights and biases

# Repeat for given number of epochs
for i in range(1000):

    # 1. Generate predictions
    preds = model(inputs)

    # 2. Calculate loss
    loss = mse(preds, targets)

    # 3. Compute gradients
    loss.backward()

    # 4. Update parameters using gradients
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5

        # 5. Reset the gradients to zero
        w.grad.zero_()
        b.grad.zero_()

print(loss)
print(preds)