"""
这个文件中包括了对张量的求导操作
"""

import torch


# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3
# average = z.mean()
#
# # print(z, average)
# average.backward()
# print(x.grad)

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = torch.pow(x, 2) + torch.pow(y, 3)

z.backward()    # 求导

print(x.grad)
print(y.grad)
# 输出导数

# 标量对向量求导
X = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
W = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
Y = torch.add(torch.dot(X, W), b)

Y.backward()
print(W.grad)
print(b.grad)


# 向量对向量求导
N = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
M = N.t().mm(N)
M.backward(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
print(N.grad)


# 高阶导数
print("高阶导数:")
x = torch.tensor(5.0, requires_grad=True)
y = torch.pow(x, 3)

grad_x = torch.autograd.grad(y, x, create_graph=True)
print(grad_x)       # dy/dx = 3 * x2，即75

grad_grad_x = torch.autograd.grad(grad_x[0], x)
print(grad_grad_x)  # 二阶导数 d²y/dx = 30
