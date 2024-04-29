import torch

# 标量(即它包含一个元素数据)
# x = torch.ones(3, 4, requires_grad=True)
# print(x)
# y = 2*x + 3  # y = x[1][1] + 2
# # print(x.mean())  # 取平均
#
# z = y * y * 2  # z = (x+2)(x+2)*3 = 3(x^2) + 12x + 12
#
# out = z.mean()
#
# print(z)
# print(out)
#
# out.backward()
# print(x.grad)  # z对x求导，再处于z的维度


# 雅克比矩阵
# x = torch.randn(3, requires_grad=True)
#
# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2
#
# print(y)
# v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(v)
#
# print(x.grad)