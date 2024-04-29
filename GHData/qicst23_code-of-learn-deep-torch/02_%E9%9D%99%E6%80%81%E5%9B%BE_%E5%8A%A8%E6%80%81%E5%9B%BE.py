#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 22:24:23 2018

@author: jangqh
"""
import torch
from torch.autograd import Variable
x = Variable(torch.randn(1,10))
prev_h = Variable(torch.randn(1,20))
W_h = Variable(torch.randn(20,20))
W_x = Variable(torch.randn(20,10))

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h
next_h = next_h.tanh()

#next_h.backward(torch.ones(1, 20))


first_counter = torch.Tensor([0])
second_counter = torch.Tensor([10])

while (first_counter < second_counter)[0]:
    first_counter += 2
    second_counter += 1

print(first_counter)
print(second_counter)


































