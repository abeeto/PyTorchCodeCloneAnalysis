# -*- coding: utf-8 -*-
"""
What is PyTorch?
================

It’s a Python based scientific computing package targeted at two sets of
audiences:

-  A replacement for numpy to use the power of GPUs
-  a deep learning research platform that provides maximum flexibility
   and speed

Getting Started
---------------

Tensors
^^^^^^^

Tensors are similar to numpy’s ndarrays, with the addition being that
Tensors can also be used on a GPU to accelerate computing.
"""

from __future__ import print_function
import torch

###############################################################
# Construct a 5x3 matrix, uninitialized:

x = torch.Tensor(5, 3)
y = torch.rand(5, 3)

###############################################################
# All the Tensors on the CPU except a CharTensor support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# Tensors can be moved onto GPU using the ``.cuda`` function.

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    print("Here is in GPU!")
    x = x.cuda()
    y = y.cuda()
    print(x)
    print (y)
    x + y
    print(x+y)