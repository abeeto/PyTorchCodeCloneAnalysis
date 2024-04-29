#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:11:14 2019

@author: said
"""

import torch
import time

class MyReLU(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
#        print(ctx.saved_tensors)
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

N, D_in, H, D_out = 64,1000,100,10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True) #100 x 10

learning_rate = 1e-6

start = time.time()
for t in range(1000):
#    h = x.mm(w1) #64 x 100
#    h_relu = h.clamp(min=0) #64 x 100
#    y_pred = h_relu.mm(w2) #64 x 10
    relu = MyReLU.apply
#    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    y_pred = relu(x.mm(w1)).mm(w2)
    
#    loss = np.square(y_pred-y).sum()
    loss = (y_pred-y).pow(2).sum()
    print(t,loss.item())
    

    
#    grad_y_pred = 2.0 * (y_pred-y) #64 x 10
#    grad_w2 = h_relu.t().mm(grad_y_pred) #100 x 10
#    grad_h_relu = grad_y_pred.mm(w2.t()) #64 x 100
#    grad_h = grad_h_relu.clone()
#    grad_h[h < 0] = 0
#    grad_w1 = x.t().mm(grad_h)
    
    loss.backward()
    
#    w1 -= learning_rate*grad_w1
#    w2 -= learning_rate*grad_w2
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        
        w1.grad.zero_()
        w2.grad.zero_()

end = time.time()
print(end-start)