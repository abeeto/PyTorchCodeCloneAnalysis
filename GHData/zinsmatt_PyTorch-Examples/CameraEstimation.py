#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""


import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt



a = torch.tensor([5.0, 3.0, 2.0], requires_grad=True)

b = a.div(a[0])

print(b)

b.backward(a)

print(a.grad)

#%%
N = 10
pts = np.random.random((3, N))
pts[2, :] += 3

K = np.array([[520.9, 0.0, 325.1],
              [0.0, 521.0, 249.7],
              [0.0, 0.0, 1.0]])
uvs = K @ pts
for i in range(uvs.shape[1]):
    uvs[:, i] /= np.linalg.norm(uvs[:, i])
#uvs /= uvs[2, :]
uvs_n = uvs / uvs[2, :]


#def myLoss(x, y):
#    temp = x.clone()
#    temp[:, 0] /= temp[:, 2]
#    temp[:, 1] /= temp[:, 2]
#    temp[:, 2] /= temp[:, 2]
#    l = nn.MSELoss()
#    return l(temp, y)

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.lin = nn.Linear(3, 3, bias=False)        
    def forward(self, x):
        x = self.lin(x)
#        x[:, 0] /= x[:, 2]
#        x[:, 1] /= x[:, 2]
#        x[:, 2] /= x[:, 2]
        #x = f.normalize(x, p=2, dim=1)

        return x
    
def myLoss(x, y):
    l = nn.MSELoss()
    a = f.normalize(x, p=2, dim=1)
    b = f.normalize(y, p=2, dim=1)
    print(a)
    print(b)
    return l(a, b) #f.normalize(x, p=2, dim=1), f.normalize(y, p=2, dim=1))
    
network = Projection() # nn.Linear(3, 3)
loss_fn = myLoss #nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

for i in range(500):
    inputs = torch.Tensor(pts.T)
    labels = torch.Tensor(uvs.T)
    outputs = network(inputs)
    loss = loss_fn(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(loss.item())
print("Final loss: ", loss.item())

network.eval()
#%%

K_est = network.lin.weight.detach().numpy()
#bias = network.lin.bias.detach().numpy()
#K_est /= K_est[2, 2]
#print("K estimated : \n", K_est)


uvs_est = (K_est @ pts).T
print(uvs_est)

#uvs_est /= uvs_est[2, :]

#print(uvs_n.T)
#print(uvs_est.T)

#print(loss_fn(torch.Tensor(uvs_est.T), torch.Tensor(uvs.T)))

res = network(torch.Tensor(pts.T)).detach().numpy()
print(res)

print(uvs.T)

        
#%%
#tmp_data = np.array([[1.0, 0.0, 0.0]])
#tmp_data_t = torch.Tensor(tmp_data)
#
#print((K_est @ tmp_data.T).T)
#print(network(tmp_data_t))

#uvs_est_norm = f.normalize(torch.Tensor(uvs_est), p=2, dim=1)
#resprint(f.normalize(torch.Tensor(res), p=2, dim=1))


def normalize(data):
    for d in data:
        d /= np.linalg.norm(d)
    return data
def normalize1(data):
    for d in data:
        d /= d[2]
    return data

print(normalize1(uvs_est))
print(normalize1(res))
print(uvs_n.T)