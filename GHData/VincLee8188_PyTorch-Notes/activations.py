import torch
import torch.nn as nn
from torch.autograd import gradcheck


class LinearFunction(torch.autograd.Function):
    # Question1: about .mm() amd .t() why 2 dim, what about higher dimension input?
    # ANSWER: this function limits input to be 2-d with (N, C), which is different from nn.Linear.

    # Question2: Why Bias is only a scalar in this case?
    # Question3: What is the shape of grad_output? Why loss.backward() doesn't use backward() function in `LinearFunction`?
    # Figure out the dimension of each tensor is important for understanding!
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    # Reference
    - https://pytorch.org/docs/1.2.0/autograd.html#torch.autograd.Function
    - https://pytorch.org/docs/1.2.0/notes/extending.html#
    - https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """
    @staticmethod
    def forward(ctx, in_data, weight, bias=None):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(in_data, weight, bias)
        output = in_data.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        It must accept a context ctx as the first argument, followed by as many outputs did forward() return,
        and it should return as many tensors, as there were inputs to forward().

        Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the corresponding input.
        """
        in_data, weights, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weights)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), in_data)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, in_data):
        return LinearFunction.apply(in_data, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    """
    gradcheck takes a tuple of tensors as input, check if your gradient
    evaluated with these tensors are close enough to numerical
    approximations and returns True if they all verify this condition.
    
    # Reference
    - https://pytorch.org/docs/1.2.0/autograd.html#numerical-gradient-checking
    """
    linear = LinearFunction.apply
    input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),
             torch.randn(30, 20, dtype=torch.double, requires_grad=True))
    test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
    print(test)

    model_1 = Linear(3, 5)
    model_2 = nn.Linear(3, 5)
    print(model_1, model_2)
    print([(x.name, x.shape) for x in model_1.parameters()])
    print([(x.name, x.shape) for x in model_2.parameters()])
    # Input below work for nn.Linear but not for Linear defined in this script.
    # x = torch.randn(2, 10, 3)
    # y_real = torch.ones(2, 10, 5)
    x = torch.randn(10, 3)
    y_real = torch.ones(10, 5)
    y_pred = model_1(x)
    print(y_pred.shape)
    loss = ((y_pred - y_real)**2).sum()
    loss.backward()
    print(model_1.weight.grad.shape)
