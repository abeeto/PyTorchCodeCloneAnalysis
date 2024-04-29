#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       2_Variable.py
@software:   pycharm
Created on   7/4/18 9:11 PM

"""

import torch as tc
from torch.autograd import Variable


tensor = tc.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)
t_out = tc.mean(tensor * tensor)
v_out = tc.mean(variable * variable)

# v_out = 1/4*(var*var)
# d(v_out)/d(var) = 1/2*(var)

v_out.backward()

print t_out.requires_grad
print variable.grad
print variable.data