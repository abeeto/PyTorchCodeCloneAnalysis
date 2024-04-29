import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

w1 = torch.Tensor([-1.0])
w1.requires_grad = True
w2 = torch.Tensor([4.0])
w2.requires_grad = True
b = torch.Tensor([5.0])
b.requires_grad = True
lr = 0.005
epochs = 1000


def make_data():
    x_data = np.arange(1.0, 5.0, 1.0)
    y_data = x_data ** 2 + x_data * 2 + 1
    return x_data, y_data


def forward(x):
    return w1 * (x**2) + w2 * x + b


def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2


def train():
    x_data, y_data = make_data()
    epoch_list = []
    loss_list = []
    for epoch in range(epochs):
        for x, y in zip(x_data, y_data):
            l = loss(x, y)
            l.backward()

            w1.data -= (lr * w1.grad.data)
            w2.data -= (lr * w2.grad.data)
            b.data -= (lr * b.grad.data)
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            b.grad.data.zero_()
        print('{:>3}epoch, w1={:3.3}, w2={:3.3}, b={}, loss={:3.3}'.format(epoch+1, w1.data.item(), w2.data.item(),
                                                                               b.data.item(), l.item()))
        epoch_list.append(epoch)
        loss_list.append(l.item())

    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def infer(x):
    return w1 * (x**2) + w2 * x + b


if __name__ == "__main__":
    train()
    print(infer(7.0).data.item())
