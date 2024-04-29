#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:11:14 2019

@author: said
"""

import numpy as np
import time

N, D_in, H, D_out = 64,1000,100,10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H,D_out) #100 x 10

learning_rate = 1e-6

start = time.time()
for t in range(1000):
    h = x.dot(w1) #64 x 100
    h_relu = np.maximum(h,0) #64 x 100
    y_pred = h_relu.dot(w2) #64 x 10
    
    loss = np.square(y_pred-y).sum()
    print(t,loss)
    
    grad_y_pred = 2.0 * (y_pred-y) #64 x 10
    grad_w2 = h_relu.T.dot(grad_y_pred) #100 x 10
    grad_h_relu = grad_y_pred.dot(w2.T) #64 x 100
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

end = time.time()
print(end-start)