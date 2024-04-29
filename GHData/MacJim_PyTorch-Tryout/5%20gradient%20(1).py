import torch


# MARK: Do some simple calculations
# 2 by 2 "ones" matrix.
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad)    # None
print(x.grad_fn)    # None

y = x + 2
print(y)
print(y.grad)    # None
print(y.grad_fn)    # AddBackward0 object

z = y * y * 3
print(z)
print(z.grad)    # None
print(z.grad_fn)    # MulBackward0 object

meanZ = z.mean()
print(meanZ)    # A scalar of 27.
print(meanZ.grad)    # None
print(meanZ.grad_fn)    # MeanBackward0 object

sumZ = z.sum()
print(sumZ)    # A scalar of 108
print(sumZ.grad)    # None
print(sumZ.grad_fn)    # SumBackward0 object


# MARK: Backpropagate
# Calculate d(meanZ) / dx
meanZ.backward(retain_graph=True)    # FIXME: The `retain_graph=True` doesn't seem to work.
print(x.grad)    # 2 by 2 matrix with value 4.5

meanZ.backward(y, retain_graph=True)
print(y.grad)    # None
print(z.grad)    # None
