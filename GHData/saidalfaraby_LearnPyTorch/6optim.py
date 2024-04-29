#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:11:14 2019

@author: said
"""

import torch
import time


dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

N, D_in, H, D_out = 64,1000,100,10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out)
        )
    

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

start = time.time()
for t in range(1000):
#    relu = MyReLU.apply
#    y_pred = relu(x.mm(w1)).mm(w2)
    y_pred = model(x)
#    loss = (y_pred-y).pow(2).sum()
    loss = loss_fn(y_pred,y)
    print(t,loss.item())
    
#    model.zero_grad()
    optimizer.zero_grad()
    

    loss.backward()
    

#    with torch.no_grad():
#
#        for param in model.parameters():
#            param -= learning_rate*param.grad
    optimizer.step()

end = time.time()
print(end-start)