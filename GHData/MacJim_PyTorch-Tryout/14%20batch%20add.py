import torch


def test1():
    x = torch.Tensor([[1, 1], [1, 1]])
    y = torch.Tensor([2, 2])
    x += y

    print(x)    # tensor([[3., 3.], [3., 3.]])


def test2():
    x = torch.Tensor([[1, 1], [1, 1]])
    y = torch.Tensor([2, 2])
    y += x    # RuntimeError: output with shape [2] doesn't match the broadcast shape [2, 2]

    print(y)
