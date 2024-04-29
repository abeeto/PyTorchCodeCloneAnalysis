import torch
from torch.autograd import Variable
import numpy as np


def simple_gradient():
    torch.set_default_tensor_type('torch.DoubleTensor')
    array = np.zeros((2, 2), float)
    array[:][:] = [[1, 2], [2, 1]]
    a = torch.from_numpy(array)

    array2 = np.zeros((2, 2), float)
    array2[:][:] = [[2, 2], [2, 2]]
    b = torch.from_numpy(array2)
    # evaluates the gradient of 2*(x*x) + 5*x, i. e.
    # 4*x+5
    # at point
    # x=a
    print(type(a), " - a")
    x = Variable(a, requires_grad=True)
    print(type(x), " - x")
    z = 2 * (x * x) + 5 * x + b
    # run the backpropagation
    array3 = np.zeros((2, 2), float)
    array3[:][:] = [[2, 2], [2, 2]]
    # array3 = np.zeros((1,1), float)
    # array3[:][:] = [[1]]

    z.backward(torch.from_numpy(array3))
    # z.backward()
    print(x.data)
    print(x.grad)
    print(z.data)


def simple_gradient2():
    torch.set_default_tensor_type('torch.DoubleTensor')
    array = np.zeros((2, 2), float)
    array[:][:] = [[1, 2], [2, 1]]
    a = Variable(torch.from_numpy(array), requires_grad=True)

    array2 = np.zeros((2, 2), float)
    array2[:][:] = [[2, 2], [2, 2]]
    b = Variable(torch.from_numpy(array2), requires_grad=False)
    z = a + b
    array3 = np.zeros((2, 2), float)
    array3[:][:] = [[1, 1], [1, 1]]

    z.backward(torch.from_numpy(array3))
    print(a.grad)
    print(b.grad)


def simple_gradient3():
    torch.set_default_tensor_type('torch.DoubleTensor')
    array = np.zeros((2, 2), float)
    array[:][:] = [[1, 2], [2, 1]]
    a = Variable(torch.from_numpy(array), requires_grad=True)

    b = a * a
    z = 5 + b
    array3 = np.zeros((2, 2), float)
    array3[:][:] = [[1, 1], [1, 1]]

    z.backward(torch.from_numpy(array3))
    print(a.grad)
    # b.backward(torch.from_numpy(array3))
    print(b.grad)
    print(b.data)
    print(z.data)


def simple_gradient4():
    torch.set_default_tensor_type('torch.DoubleTensor')
    array_x1 = np.zeros((1), float)
    array_x1[:] = [1]
    x1 = Variable(torch.from_numpy(array_x1), requires_grad=True)

    array_x2 = np.zeros((1), float)
    array_x2[:] = [1]
    x2 = Variable(torch.from_numpy(array_x2), requires_grad=True)

    x1x2 = x1 * x2

    x1x1 = x1 * x1

    z = x1x2 + x1x1
    array3 = np.zeros((1), float)
    array3[:][:] = [1]
    z.backward(torch.from_numpy(array3))
    print(x1.grad)
    # b.backward(torch.from_numpy(array3))
    print(x2.grad)

    print(x1x2.grad, x1x2.requires_grad)

    print(x1x2.data)
    print(x1x1.data)


if __name__ == "__main__":
    simple_gradient4()
