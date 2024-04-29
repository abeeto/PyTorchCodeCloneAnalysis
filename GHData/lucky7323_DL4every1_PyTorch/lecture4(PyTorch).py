import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initialization
# Because optimal solution is (w1:2, w2:0, b:0),
# to initialize similar to above converges fast
w1 = Variable(torch.Tensor([0.0]), requires_grad=True)
w2 = Variable(torch.Tensor([0.0]), requires_grad=True)
b = Variable(torch.Tensor([0.0]), requires_grad=True)
alpha = 0.01

def forward(x, w1, w2, b):
    y_hat = x*x*w2 + x*w1 + b
    return y_hat

def loss(y, y_hat):
    return ((y - y_hat) * (y - y_hat))

for epoch in range(1000):
    print("\nepoch: ", epoch)
    for x_val, y_val in zip(x_data, y_data):
        l = loss(y_val, forward(x_val, w1, w2, b))
        l.backward()
        # update weights simultaneously by using temporary variables
        w1_tmp = w1.data - alpha * w1.grad.data
        w2_tmp = w2.data - alpha * w2.grad.data
        b_tmp = b.data - alpha * b.grad.data
        w1.data = w1_tmp
        w2.data = w2_tmp
        b.data = b_tmp
        l = loss(y_val, forward(x_val, w1, w2, b))
        print("w1: %.2f, w2: %.2f, b: %.2f ====> loss: %.3f" % (w1.data, w2.data, b.data, l))
        # manually zero the gradients after updating weights
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("prediction of 4 hours: %.2f" % forward(4, w1, w2, b))
