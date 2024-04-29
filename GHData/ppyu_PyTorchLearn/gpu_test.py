# -*- coding: utf-8 -*-
"""
@File   : gpu_test.py
@Author : Pengy
@Date   : 2020/10/12
@Description : 测试GPU环境是否可用
"""
import torch

if __name__ == '__main__':
    # 测试CUDA
    print("Support CUDA ?:", torch.cuda.is_available())
    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)

    y = torch.randn(2, 3)
    y = y.cuda()
    print(y)

    z = x + y
    print(z)

    # 测试CUDNN
    from torch.backends import cudnn

    print("Support cudnn ?:", cudnn.is_available())
