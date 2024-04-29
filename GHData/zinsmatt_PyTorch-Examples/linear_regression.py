#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:05:23 2019

@author: matt
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


in_size = 1
out_size = 1
n_epochs = 60
lr = 0.001


x = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)




model = nn.Linear(in_size, out_size)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for e in range(n_epochs):
    inputs = torch.Tensor(x)
    targets = torch.Tensor(y)

    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print(e, loss.item())

params = list(model.parameters())
a = params[0].data
b = params[1].data

xmin = x.min()
xmax = x.max()

plt.plot([xmin, xmax], [a*xmin+b, a*xmax+b], "r")
plt.scatter(x, y)
plt.show()
