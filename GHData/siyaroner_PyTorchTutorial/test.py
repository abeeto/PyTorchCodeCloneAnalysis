import torch
x = torch.randn(3, requires_grad=True)

y = x ** 2
for _ in range(10):
    y = y * 2

    print(y)
print(y.shape)
# y.backward(x)
print(x.grad)