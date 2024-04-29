# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:50:32 2020

@author: shankarj
"""

import torch as pt
import numpy as np
import matplotlib.pyplot as plt

class LinearReg(pt.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linearMod = pt.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        pred = self.linearMod(x)
        return pred
    
    def get_params(self):
        [w, b] = self.linearMod.parameters()
        return (w[0][0].item(), b[0].item())
    

#Create some linear data
x = pt.randn(100, 1) * 18
y = x + (5 * pt.randn(100, 1))

lr = LinearReg(1, 1)
w_init, b_init = lr.get_params()

objective = pt.nn.MSELoss();
optimizer = pt.optim.SGD(lr.parameters(), lr = 0.003)
epochs = 100
loss_history = []

for i in range(epochs):
    y_pred = lr.forward(x)
    loss = objective(y_pred, y)
    print(f'Epoch {i+1} : Loss value {loss}')
    loss_history.append(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

w_final, b_final = lr.get_params()

data_range = np.array([min(x).item(), max(x).item()])
initial_fit_line = w_init * data_range + b_init
final_fit_line = w_final * data_range + b_final

plt.plot(x.numpy(), y.numpy(), 'o', label='data')
plt.plot(data_range, initial_fit_line, 'b', label='init_fit')
plt.plot(data_range, final_fit_line, 'r', label='final_fit_gd')
plt.legend()
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()




        