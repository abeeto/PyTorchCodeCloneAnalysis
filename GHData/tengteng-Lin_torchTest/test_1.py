import torch

x = torch.ones(3,5,requires_grad=True) #requires_grad会跟踪x的计算
# print(x.grad) #None
y = x + 2
y.requires_grad_(True)
# print(y.grad_fn)
z = y * y * 3
out = z.mean() #out是标量
# print(z,out)

out.backward()
print(x.grad) #out对x的导数

#雅克比向量积例子
# x = torch.tensor([2.0,3.0],requires_grad=True)
# y = x * x * 2
# print(y)  #y不再是标量
#
#
# v = torch.tensor([1,1.0])
# y.backward(v)
# print(x.grad)  #是x，为啥是x？？