# Are torch variables assigned by reference?

import torch


def test1():
    a = torch.Tensor(list(range(8))).reshape(2, -1)
    b = a
    b[0, 0] = 21
    # `a` and `b` refer to the same object and are both changed!
    # So torch uses native Python reference mechanics.
    print("a:", id(a), a)
    print("b:", id(b), b)


def test2():
    a = torch.Tensor(list(range(8))).reshape(2, -1)
    b = a
    b = b * 2
    # `a` remain unchanged.
    print("a:", id(a), a)
    print("b:", id(b), b)


def test3():
    a = torch.Tensor(list(range(8))).reshape(2, -1)
    b = a
    b *= 2
    # `a` and `b` refer to the same object and are both changed!
    print("a:", id(a), a)
    print("b:", id(b), b)


def test4():
    a = torch.Tensor(list(range(8))).reshape(2, -1)
    # MARK: Somehow torch tensors use `clone` instead of the sandard `copy` function.
    b = a.clone()
    b *= 2
    # `a` remain unchanged.
    print("a:", id(a), a)
    print("b:", id(b), b)


test1()
test2()
test3()
test4()
