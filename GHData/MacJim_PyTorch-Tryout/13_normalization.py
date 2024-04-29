"""
Tests Torch's normalization functions.
"""

import random

import torch
import torch.nn.functional as F


# MARK: - L* normalization
def test_L1_normalization_1():
    x1 = torch.tensor([1, 1, 1, 5], dtype=torch.float32)
    print(f"x1: {x1}")    # tensor([1., 1., 1., 5.])
    print(f"L1 normalized: {F.normalize(x1, p=1, dim=0)}")    # tensor([0.1250, 0.1250, 0.1250, 0.6250])


def test_L2_normalization_1():
    x1 = torch.tensor(list(range(10)), dtype=torch.float32)
    print(f"x1: {x1}")    # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    print(f"L2 normalized: {F.normalize(x1, p=2, dim=0)}")    # tensor([0.0000, 0.0592, 0.1185, 0.1777, 0.2369, 0.2962, 0.3554, 0.4146, 0.4739, 0.5331])

    x2 = torch.tensor([4] * 8, dtype=torch.float32)
    x2 = x2.view(4, -1)
    print(f"x2: {x2}")    # ([[4., 4.], [4., 4.], [4., 4.], [4., 4.]])
    print(f"L2 normalized dim=0: {F.normalize(x2, p=2, dim=0)}")    # tensor([[0.5000, 0.5000], [0.5000, 0.5000], [0.5000, 0.5000], [0.5000, 0.5000]]) That is, normalize [4 4 4 4]: 1 / 2 = 4 / sqrt(16 * 4)
    print(f"L2 normalized dim=1: {F.normalize(x2, p=2, dim=1)}")    # tensor([[0.7071, 0.7071], [0.7071, 0.7071], [0.7071, 0.7071], [0.7071, 0.7071]]) That is, normalize [4 4]: sqrt(2) / 2 = 4 / (4 * sqrt(2)) = 4 / sqrt(16 + 16).


def test_L2_normalization_2():
    """
    Can only normalize torch tensors.
    Integer tensors are not supported.
    """
    x1 = torch.tensor(list(range(10)))
    print(f"x1: {x1}")    # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"L2 normalized: {F.normalize(x1, p=1, dim=0)}")    # RuntimeError: Can only calculate the mean of floating types. Got Long instead.


# MARK: - Batch norm
def test_batch_norm_1():
    x1 = torch.Tensor(list(range(-20, 20)))
    x1 = x1.view(4, -1)    # Simulates 4 batches, each containing 10 elements.
    print("x1:", x1)

    x2 = torch.Tensor(list(range(40)))
    x2 = x2.view(4, -1)
    print("x2:", x2)

    x3 = list(range(40))
    random.shuffle(x3)
    x3 = torch.Tensor(x3)
    x3 = x3.view(4, -1)
    print("x3:", x3)

    # Normalize the element of different batches.
    batch_norm = torch.nn.BatchNorm1d(x1.shape[1])
    # batch_norm = torch.nn.BatchNorm1d(10)

    y1 = batch_norm(x1)
    print("y1:", y1)
    mean1 = torch.mean(y1, dim=0)
    print("Mean 1:", mean1)

    y2 = batch_norm(x2)
    print("y2:", y2)    # The same with y1.
    mean2 = torch.mean(y2, dim=0)
    print("Mean 2:", mean2)

    y3 = batch_norm(x3)
    print("y3:", y3)
    mean3 = torch.mean(y3, dim=0)
    print("Mean 3:", mean3)


# MARK: - Main
if (__name__ == "__main__"):
    test_L1_normalization_1()
    # test_L2_normalization_1()
    # test_L2_normalization_2()

    # test_batch_norm_1()
