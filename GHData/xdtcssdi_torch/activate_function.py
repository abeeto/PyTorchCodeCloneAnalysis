import matplotlib.pyplot as plt
import torch

'''
关于detach的内容
https://www.cnblogs.com/jiangkejie/p/9981707.html

返回一个不计算梯度的Variable

'''


def xyplot(x_vals, y_vals, name, idx):
    plt.subplot(3, 2, idx)
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')

if __name__ == '__main__':
    x = torch.arange(-10, 10, 0.1, requires_grad=True)

    x.relu().sum().backward()
    xyplot(x, x.detach().relu(), "relu", 1)
    xyplot(x, x.grad, "relu grad", 2)
    x.grad.zero_()

    x.sigmoid().sum().backward()
    xyplot(x, x.detach().sigmoid(), "sigmoid", 3)
    xyplot(x, x.grad, "sigmoid grad", 4)
    x.grad.zero_()

    x.tanh().sum().backward()
    xyplot(x, x.detach().tanh(), "tanh", 5)
    xyplot(x, x.grad, "tanh grad", 6)
    x.grad.zero_()

    plt.show()