import torch


# MARK: - Equal
# MARK: Tensor and tensor
def test_equal_1():
    x1 = torch.Tensor([1.0])
    x2 = torch.Tensor([1.0])
    print(id(x1) == id(x2), id(x1), id(x2))    # False
    print(x1 == x2)    # tensor([True])
    print(x1.dtype, x2.dtype)    # torch.float32 torch.float32


def test_equal_2():
    """
    Tensor comparison is per-element and results in tensors.
    """
    x1 = torch.Tensor([1.0, 2.0])
    x2 = torch.Tensor([1.0, 2.0])
    print(id(x1) == id(x2), id(x1), id(x2))    # False
    print(x1 == x2)    # tensor([True, True])
    print(x1.dtype, x2.dtype)    # torch.float32 torch.float32


def test_equal_3():
    x1 = torch.Tensor([1.0, 2.0])
    x2 = torch.Tensor([1.0, 3.0])
    print(x1 == x2)    # tensor([ True, False])


def test_equal_4():
    x1 = torch.Tensor([1.0, 2.0, 3.0])
    x2 = torch.Tensor([1.0])
    x3 = torch.Tensor([2.0])
    print(x1 == x2)    # tensor([ True, False, False])
    print(x1 == x3)    # tensor([False,  True, False])


def test_equal_5():
    x1 = torch.Tensor(list(range(4)))
    x2 = torch.Tensor([1.0, 2.0])
    print(x1 == x2)    # RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 0


def test_equal_6():
    """
    `torch.Tensor` is an alias for the default tensor type (`torch.FloatTensor`).
    """
    x1 = torch.Tensor([1.0, 2.0])
    x2 = torch.Tensor([1, 2])    # This one is float32 by default.
    print(x1 == x2)    # tensor([True, True])
    print(x1.dtype, x2.dtype)    # torch.float32 torch.float32


def test_equal_7():
    """
    - Use lowercase `torch.tensor` to create non-float tensors.
    - `float32` and `int64` can equal.
    """
    x1 = torch.tensor([1.0, 2.0])
    x2 = torch.tensor([1, 2], dtype=torch.long)
    print(x1 == x2)    # tensor([True, True])
    print(x1.dtype, x2.dtype)    # torch.float32 torch.int64


# MARK: Tensor and list
def test_equal_11():
    """
    Cannot compare tensor with list.
    """
    x1 = torch.tensor([1.0, 2.0])
    x2 = [1.0, 2.0]    # False
    print(x1 == x2)


# MARK: Tensor and value
def test_equal_21():
    x1 = torch.Tensor(list(range(15)))
    print(x1 == 2.0)    # tensor([False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False])


# MARK: - Larger or smaller than
def test_larger_smaller_1():
    x1 = torch.Tensor([1.0])
    x2 = torch.Tensor([2.0])
    x3 = torch.Tensor([3.0])

    print(x1)
    print(x2)
    print(x3)

    print(x1 < x2)    # tensor([True])
    print(x2 > x3)    # tensor([False])
    print(x3 > x2)    # tensor([True])


def test_larger_smaller_2():
    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([2.0, 3.0, 4.0])
    x3 = torch.tensor([2.0, 2.0, 2.0])

    print(x1 < x2)    # tensor([True, True, True])
    print(x1 < x3)    # tensor([ True, False, False])
    print(x2 < x3)    # tensor([False, False, False])


# MARK: Sum up
def test_larger_smaller_11():
    """
    We can sum up `True` values in `bool` tensors.
    """
    x1 = torch.tensor(list(range(16)))

    print(x1 < 6)    # tensor([ True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False])
    print((x1 < 6).sum())    # tensor(6)

    print(x1 < 10)    # tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False])
    print((x1 < 10).sum())    # tensor(10)


test_larger_smaller_11()
