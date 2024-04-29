from torch.autograd import Function, Variable

import test_cpp
import torch


class TestFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        return test_cpp.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = test_cpp.backward(gradOutput)
        return gradX, gradY


class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()

    def forward(self, inputA, inputB):
        return TestFunction.apply(inputA, inputB)


x = Variable(torch.Tensor([1, 2, 3]), requires_grad=True)
y = Variable(torch.Tensor([4, 5, 6]), requires_grad=True)
test = Test()
z = test(x, y)
z.sum().backward()
print('x: ', x)
print('y: ', y)
print('z: ', z)
print('x.grad: ', x.grad)
print('y.grad: ', y.grad)
