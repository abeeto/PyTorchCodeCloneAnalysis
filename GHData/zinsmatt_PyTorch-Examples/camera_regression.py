#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





pts = np.random.random((3, 100))
pts[2, :] += 3

K = np.array([[250, 0, 300],
              [0, 255, 250],
              [0.0, 0.0, 1.0]])


pts_n = pts / pts[2, :] # on the image plane (z=1)
uvs = K @ pts_n



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.K = nn.Linear(3, 3, bias=False)
        
    def forward(self, x):
        return self.K(x)
    
    
    
def MyLoss(x, y):
#    print("x = ", x)
#    print("y = ", y)
    l = nn.MSELoss()
#    x_n = x / x[:, 2].view((-1, 1))
#    print(x_n)
#    print(y)
    l1 = nn.MSELoss()
    return l(x[:, :2], y[:, :2]) + l1(network.lin.weight[2, :], torch.tensor([0.0, 0.0, 1.0])) 


network = Network()
optimizer = torch.optim.SGD(network.parameters(), lr=1)

loss_fn = MyLoss

# nn.init.uniform_(network.lin.weight)

# network.K.weight.data[0, 0] = 100.0
# network.K.weight.data[0, 1] = 0.0
# network.K.weight.data[0, 2] = 50.0
# network.K.weight.data[1, 0] = 0.0
# network.K.weight.data[1, 1] = 100.0
# network.K.weight.data[1, 2] = 50.0
# network.K.weight.data[2, 0] = 0.0
# network.K.weight.data[2, 1] = 0.0
# network.K.weight.data[2, 2] = 0.0



temp_results = []

for i in range(10000):
    x = torch.Tensor(pts_n.T)
    y = torch.Tensor(uvs.T)
    
    output = network(x)
    
    optimizer.zero_grad()
    
    loss = F.mse_loss(output, y)
    
    loss.backward()
    optimizer.step()
    
    print(loss.item())
    if i % 50 == 0:
        temp_results.append(network.K.weight.detach().numpy().copy())
    
K_est = network.K.weight.detach().numpy()

temp_results.append(K_est)

#%%
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


for i in range(len(temp_results)):
    K_est = temp_results[i]
    outputs = K_est @ pts_n
    plt.scatter(uvs[0, :], uvs[1, :], c='r')
    plt.scatter(outputs[0, :], outputs[1, :], marker='+', c='b')
    plt.savefig("output/result_optim_%05d.png" % i)
    plt.close()
