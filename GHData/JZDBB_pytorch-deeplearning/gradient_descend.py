import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value

def loss(y_pred, y):
    return (y_pred-y) * (y_pred-y)

print("predict (before training)",  4, 4*w.data[0])

for echo in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = w * x_val
        l = loss(y_pred, y_val)
        l.backward()
        print(x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01*w.grad.data[0]

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
    print("progress:", echo, l.data[0])

print(4, (4 * w).data[0])