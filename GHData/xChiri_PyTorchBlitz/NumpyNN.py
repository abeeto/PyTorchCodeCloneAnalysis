import numpy as np
import torch

class MyReLU(torch.autograd.Function):
    """
    We can implement our custom autograd Functions by subclassing torch.autograd.Function and
    implementing the forwards and backward passes which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return a Tensor containing the output.
        ctx is a context object that can be used to stash information for backward computation. You can cache
        arbitrary objects for use in backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output,
        and we need to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

device = torch.device("cpu")
# device = torch.device("cuda:0") # uncomment this line to run on GPU

# N is the batch size, D_in is the input dimension, H is the hidden dimension, D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold input and output data
# setting requires_grad=False indicates that we do not need to compute gradients with respect to these
# Tensors during the backward pass
x = torch.randn(N, D_in, device=device, dtype=torch.float)
y = torch.randn(N, D_out, device=device, dtype=torch.float)

# randomly initialize Tensors for weights
# setting requires_grad=True indicates that we want to compute gradients with respect to these
# Tensors on the backward pass
w1 = torch.randn(D_in, H, device=device, dtype=torch.float, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=torch.float, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # to apply our autograd functions, we use the Function.apply method.
    relu = MyReLU.apply

    # forward pass: compute predicted y using operations on Tensors; we do not need to hold intermediate
    # values since we are not implementing the backward pass by hand
    y_pred = relu(x.mm(w1)).mm(w2)

    # compute and print loss using operations on Tensors
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # backprop to compute gradients of w1 and w2 with respect to loss
    # grad_y_pred = 2.0 * (y_pred - y)
    # grad_w2 = h_relu.t().mm(grad_y_pred)
    # grad_h_relu = grad_y_pred.mm(w2.t())
    # grad_h = grad_h_relu.clone()
    # grad_h[h < 0] = 0
    # grad_w1 = x.t().mm(grad_h)

    # use autograd to compute the backward pass. This call will compute the gradient of loss
    # with respect to all Tensors with requires_grad=True. After this, w1.grad and w2.grad
    # will be Tensors holding the gradient of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # manually update weights using gradient descent. Wrap in torch.no_grad() because weights have
    # requires_grad=True, but we do not need to keep track of this in autograd.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero the gradients after updating the weights
        w1.grad.zero_()
        w2.grad.zero_()