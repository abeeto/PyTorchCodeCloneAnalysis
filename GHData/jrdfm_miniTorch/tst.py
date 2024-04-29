#!/usr/local/bin/python

from mytorch.tensor import Tensor
from mytorch.nn.module import BatchNorm2d,Flatten

import numpy as np
import torch
import torch.nn as nn

# BatchNorm2d test

if __name__ == "__main__":

    #BatchNorm22d

    # input = torch.randn(20, 100, 35, 45)
    # i = Tensor(input.numpy())

    # m_af = torch.nn.BatchNorm2d(100)
    # m = torch.nn.BatchNorm2d(100, affine=False)
    # m_mi = BatchNorm2d(100)

    # out_af = m_af(input)
    # out = m(input)
    # out_mi = m_mi(i)

    # print(f'out_af == out_mi {np.array_equal(out_af.detach().numpy(), out_mi.data)}  close {np.allclose(out_af.detach().numpy(), out_mi.data)}')
    # print(f'out == out_mi {np.array_equal(out.numpy(), out_mi.data)}  close {np.allclose(out.numpy(), out_mi.data)}')

    # Flatten
    in_c = np.random.randint(5,15)
    width = np.random.randint(60,80)
    batch_size = np.random.randint(1,4)

    x1d = np.random.randn(batch_size, in_c, width)
    x2d = np.random.randn(batch_size, in_c, width, width)
    x1d_tensor = Tensor(x1d, requires_grad=True)
    x2d_tensor = Tensor(x2d, requires_grad=True)
    test_model = Flatten()

    # torch_model = nn.Flatten()

    y_1_2_m = x1d_tensor.flatten()

    y_2_2_m = x2d_tensor.flatten()

    y_1_n = x1d.flatten()
    y_2_n = x2d.flatten()

    print(f'x1d_tensor shape {x1d_tensor.shape} x2d_tensor shape {x2d_tensor.shape}\n \
        x1d shape {x1d.shape} x2d shape {x2d.shape}')
    print(f'y_1_2_m shape {y_1_2_m.shape} y_2_2_m shape {y_2_2_m.shape}\n \
        y_1_n shape {y_1_n.shape} y_2_n shape {y_2_n.shape}')
