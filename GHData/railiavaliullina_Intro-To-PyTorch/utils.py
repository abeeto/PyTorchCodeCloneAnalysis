from torch.autograd import Function
import torch
import numpy as np
from torch import nn
from torch.autograd import gradcheck


"""
УЛУЧШЕНИЯ:

1) вынести в отдельную папку
2) все кастомные функции и классы вынести в custom_utils.py
3) вынести функцию логирования, принимающей на вход dict (name: value) 
4) добавить документацию к каждой функции, следить за названиями функций
5) TODO: ...
"""


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # функция, которая сохраняет входные/выходные тензоры метода forward
        # для дальнейшего использования в методе backward
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # на выходе метода forward 1 тензор, поэтому на вход методу backward
    # будет так же подаваться один тензор
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # проверка ctx.needs_input_grad - необязательна и предназначена
        # только для повышения эффективности
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # на входе метода forward 3 тензора, поэтому на выходе метода backward
        # будет так же 3 тензора
        return grad_input, grad_weight, grad_bias


class ConstMultiplication(Function):
    @staticmethod
    def forward(ctx, input, const):
        ctx.constant = const
        return input*const

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.constant, None


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # инициализируем операцию
        self.linear = LinearFunction.apply
        self.mulconst = ConstMultiplication.apply
        self.constant = 0.5

        # инициализируем обучаемые параметры
        self.weights = nn.Parameter(torch.randn(10, 1024) / np.sqrt(1024))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        tensor = self.linear(xb, self.weights, self.bias)
        out = self.mulconst(tensor, self.constant)
        return out


if __name__ == '__main__':
    # TODO: вынести тест в отдельную функцию
    linear = LinearFunction.apply
    input = torch.randn(32, 100, dtype=torch.double, requires_grad=True)
    weight = torch.randn(10, 100, dtype=torch.double, requires_grad=True)
    bias = torch.randn(10, dtype=torch.double, requires_grad=True)
    assert gradcheck(linear, (input, weight, bias), eps=1e-6, atol=1e-4)

    mulconst = ConstMultiplication.apply
    const = 0.5
    assert gradcheck(mulconst, (input, const), eps=1e-6, atol=1e-4)
