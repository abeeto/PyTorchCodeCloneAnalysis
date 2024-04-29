#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       1_numpy_torch.py
@software:   pycharm
Created on   7/4/18 8:55 PM

"""

import torch as tc
import numpy as np


# numpy to torch and tensor to array
np_data = np.arange(6).reshape((2,3))
torch_data = tc.from_numpy(np_data)
tensor2array = torch_data.numpy()

print 'numpy data:\n',np_data
print 'torch_data:\n',torch_data
print 'tensor2array\n',tensor2array

a = [1]
tensor = tc.FloatTensor(a)
print tc.sin(tensor)
print np.sin(1)

# matrix multiply

data = [[1,2],[3,4]]
tensor = tc.FloatTensor(data)

print 'numpy\n:',np.matmul(data,data)
print 'torch\n:',tc.mm(tensor,tensor)
print tensor.requires_grad

