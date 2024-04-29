import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx is a context object that is used to save info for backward computation
        # You can cache objects for use inthe backward pass by using ctx.save_for_backward\
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Here we recieve a tensor containing the gradient of the loss with respect to output
        # and we need to compute the gradient of the loss with respect to the input
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")

batch_size, D_in, hidden_size, D_out = 64, 1000, 100, 10

x = torch.randn(batch_size, D_in, device=device, dtype=dtype)
y = torch.randn(batch_size, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, hidden_size, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(hidden_size, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    # This applies our function
    relu = MyReLU.apply
    y_pred = relu(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
