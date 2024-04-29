#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:59:54 2019

@author: Oleksiy Grechnyev
"""
import numpy as np
import torch

print('torch.__version__=', torch.__version__)

x = torch.randn(3, requires_grad=True)
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)
