from torch import tensor, zeros, Tensor
from torch_model.check import check


@check
def add(a: Tensor, b: Tensor) -> Tensor:
    return a + b


@check
def main():
    a = zeros([2, 3])
    b = tensor([3, 4])
    c = add(a, b)
