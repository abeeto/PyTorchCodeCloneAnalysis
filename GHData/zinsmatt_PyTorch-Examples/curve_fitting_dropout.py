#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CurveFittingNet(nn.Module):
    def __init__(self, hidden_layers):
        super(CurveFittingNet, self).__init__()
        hidden_layers = [1] + hidden_layers
        self.layers = [nn.Sequential(
                nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                nn.ReLU()) for i in range(1, len(hidden_layers))]
        self.fc = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, x):
        out = x
        for lay in self.layers:
            out = lay(out)
        return self.fc(out)
    
    
    
#%% Data preparation
        
def f(x):
    return 0.25 * np.cos(x) * x# + np.sin(-0.5*x)

training_range = [-5, 5]
eval_range = [-10, 10]

X = np.linspace(training_range[0], training_range[1], 10)
Y = f(X)

plt.scatter(X, Y)
plt.xlim(eval_range)
plt.ylim([-2.0, 2.0])


#%% Training

model = CurveFittingNet([20] * 10)
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)

# use the modules apply function to recursively apply the initialization
model.apply(init_normal)
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.051)
n_epochs = 3000

for e in range(n_epochs):
    inputs = torch.Tensor(X).unsqueeze(1)
    labels = torch.Tensor(Y).unsqueeze(1)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("epoch %d loss %f" % (e, loss))
    
#%% Eval
model.eval()
X_eval = np.linspace(eval_range[0], eval_range[1], 1000)
gt_eval = f(X_eval)

#with torch.no_grad():
inputs_eval = torch.Tensor(X_eval).unsqueeze(1)
inputs_eval.requires_grad = True
outputs_eval = model(inputs_eval)
res = outputs_eval.detach().numpy().ravel()

temp_y = torch.Tensor(gt_eval).unsqueeze(1)
loss = loss_fn(outputs_eval, temp_y)
optimizer.zero_grad()
loss.backward()




plt.scatter(X_eval, res, c="red", s=0.4)
plt.scatter(X_eval, gt_eval, c="green", s=0.6)

#plt.figure("variance")
#plt.plot(X_eval, inputs_eval.grad)