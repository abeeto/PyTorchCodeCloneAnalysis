import torch

x = torch.randn((1, 1), requires_grad=True)

w = torch.FloatTensor([[2],[3]])
y = torch.matmul(w, x)
# y = x ** 2

v = torch.FloatTensor([[1], [1]])
y.backward(v) # 当 y 不再是 scalar 需要加一个 v 帮助 backward.
print('x', x)
print('w', w)
print('y', y)
print(x.grad) #tensor([[5.]])
