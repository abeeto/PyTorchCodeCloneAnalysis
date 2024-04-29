"""
Tests the effect of padding on numerous operators.
"""

import os

import torch
import torch.nn.functional as F


def test_max_pool_1():
    """
    3x3 -> kernel 2 -> 1x1

    The final row and column are simply ignored.
    """
    x = torch.Tensor(range(9))
    x = x.reshape((1, 1, 3, 3))
    y = F.max_pool2d(x, 2)
    # y1 = y1.reshape((-1, y1.shape[-1]))
    print(f"x1: {x}")    # [[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]]
    print(f"y1: {y}")    # [[[[4.]]]]


def test_max_pool_2():
    x = torch.Tensor(range(9))
    x = x.reshape((1, 1, 3, 3))
    y = F.max_pool2d(x, 2, padding=1)
    print(f"x1: {x}")    # [[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]]
    print(f"y1: {y}")    # [[[[0., 2.], [6., 8.]]]]


# MARK: - Main
if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Working directory: {os.getcwd()}")

    # test_max_pool_1()
    test_max_pool_2()
