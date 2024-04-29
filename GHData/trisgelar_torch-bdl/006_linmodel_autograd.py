# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:18:32 2018

@author: trisgelar
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], [7.042], [10.79], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.67], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

plt.plot(x_train, y_train, 'ro', label = 'Original Data')
plt.show()

X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)

print("requires_grad for x_train", X_train.requires_grad)
print("requires_grad for y_train", Y_train.requires_grad)

input_size = 1
hidden_size = 100
output_size = 1
learning_rate = 1e-6

w1 = torch.rand(input_size, hidden_size, requires_grad=True)
w1.shape

w2 = torch.rand(hidden_size, output_size, requires_grad=True)
w2.shape

for iter in range(1, 301):
    y_pred = X_train.mm(w1.clamp(min=0).mm(w2))
    loss = (y_pred - Y_train).pow(2).sum()
    if iter % 50 == 0:
        print(iter, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
    
print("w1: {}".format(w1))
print("w2: {}".format(w2))

x_train_tensor = torch.from_numpy(x_train)
predicted_in_tensor = x_train_tensor.mm(w1).clamp(min=0).mm(w2)
predicted_in_tensor

predicted = predicted_in_tensor.detach().numpy()
predicted

plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label="Fitted Line")
plt.legend()
plt.show()




