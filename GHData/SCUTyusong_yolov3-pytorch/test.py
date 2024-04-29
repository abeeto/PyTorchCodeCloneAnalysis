import numpy as np
import matplotlib.patches as patches
from anchor import *
import torch
import torch.nn as nn

# def foo():
#     print('starting...')
#     while True:
#         res=yield 4
#         print('res:',res)
# g=foo()
# print(next(g))
# print('*'*20)
# print(next(g))
# aa=[1,2,3,4,5]
# bb=[1,2,3,4,5,6]
# print(aa(1))
# for i in range(0):

loss_fn = torch.nn.BCELoss()
input = torch.randn(1,2, 3, 4)
target = torch.randn(1,2,3, 4)
loss = loss_fn(input, target)
print(loss.shape)

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss_fn(input, target)
# print(output.shape)
