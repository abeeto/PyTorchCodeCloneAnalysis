#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       5_bulid_nn_quickly.py
@software:   pycharm
Created on   7/5/18 2:07 PM

"""
import torch as tc

net = tc.nn.Sequential(
    tc.nn.Linear(2,10),
    tc.nn.ReLU(),
    tc.nn.Linear(10,2)
)
print net