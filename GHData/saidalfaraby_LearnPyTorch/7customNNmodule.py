#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:11:14 2019

@author: said
"""

import torch
import time
import torch.nn.functional as F

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
#        super(TwoLayerNet, self).__init__()
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,D_out)
        
    def forward(self,x):
#        h_relu = self.linear1(x).clamp(min=0)
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
        


dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

N, D_in, H, D_out = 64,1000,100,10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)
    

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

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