import torch
import torch.autograd

A = torch.ones(2, 2, requires_grad=True)
B = torch.ones(2, 2, requires_grad=True)

print(A + B)

# gradients

X = torch.ones(2, requires_grad=True)

Y = 5 * (X + 1) ** 2  # 20, 20

print(Y)
Z = (1 / 2) * torch.sum(Y)  # same as Y.mean()
# to use backward we need a scalar, by doing the mean we found it
Z.backward()
print(X.grad)












