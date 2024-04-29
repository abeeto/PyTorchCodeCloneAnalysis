# -*- coding=utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃ uncomment to use gpu

# N is the numbers of input and D_in is the input size
# H is the size of hidden layer; D_out is the output size
N, D_in, H, D_out = 64, 1000, 100, 10

# create the random input and output following the normal distribution
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# create the random weight and bias following the normal distribution
# set requires_grad = True, so the gradient can be obtained in related parameters
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# set the learning rate, which can control the scale of parameter change
learning_rate = 1e-6
for t in range(500):
    # forward propogation and get the prediction values of y 
    # the computing graph is formed becasue the requires_grad is set as True
    # the gradient is calculated automately, which is one of the important tools in PyTorch
    # .mm() is to calculate the matrix multiplication
    # torch.clamp(input, min, max, out=None) is to control the limit of tensor
    # torch.clamp is similar to ReLU（Rectified Linear Unit） function
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # calculate the loss by MSE (mean square error)
    # loss is also a tensor and it can be changed into Scale by the function of .item()
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # use autograd to calculate the backward propogation 
    # all the gradients of tensor with requires_grad=True can be got
    loss.backward()

    # torch.no_grad(): update the weight and not change the computing graph
    # .grad() to get the gradient
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # clear the gradients each step when it has been used, otherwise the gradients will be accumlated
        w1.grad.zero_()
        w2.grad.zero_()