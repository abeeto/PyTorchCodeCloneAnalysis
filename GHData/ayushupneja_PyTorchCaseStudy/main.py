import torch
import matplotlib.pyplot as plt

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, graDoutput):
        input, = ctx.saved_tensors
        graDinput = graDoutput.clone()
        graDinput[input < 0] = 0
        return graDinput
dtype = torch.float
device = torch.device("cpu")

N, Din, H, Dout = 64, 1000, 100, 10

x = torch.randn(N, Din, device=device, dtype=dtype)
y = torch.randn(N, Dout, device=device, dtype=dtype)

weight1 = torch.randn(Din, H, device=device, dtype=dtype, requires_grad=True)
weight2 = torch.randn(H, Dout, device=device, dtype=dtype, requires_grad=True)
graph = list(range(0,500))

learning_rate = 1e-6
for t in range(500):
    relu = MyReLU.apply
    y_pred = relu(x.mm(weight1)).mm(weight2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    graph[t] = loss.item()
    loss.backward()
    with torch.no_grad():
        weight1 -= learning_rate * weight1.grad
        weight2 -= learning_rate * weight2.grad
        weight1.grad.zero_()
        weight2.grad.zero_()

plt.plot(graph)
plt.show()
