import torch
from torch.autograd import Variable



x_data = [11., 22., 33.]
y_data = [21., 14., 64.]

w = Variable(torch.tensor([1.]), requires_grad=True)


def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)**2

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()

    print("progress: ", epoch, l.data[0])
