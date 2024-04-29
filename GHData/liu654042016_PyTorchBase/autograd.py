import torch

x = torch.randn(3, 4, requires_grad=True)

b = torch.randn(3, 4, requires_grad=True)

t = x + b

y = t.sum()

y.backward()

x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)

y = w * x
z = y + b

w2 = torch.rand(1, requires_grad=True)
b2 = torch.rand(1, requires_grad=True)
res = w2*z + b2

#判断是否是leaf节点， 显然这个简单的公式里 y不是
print(y.is_leaf)

#retain_graph = True，保留z-y-x的计算图
z.backward(retain_graph=True)
print(w.grad)
#非叶子节点， grad直接省略
#如果需要非叶子节点的grad，需要这个函数
y.retain_grad()
print(y.grad)

res.backward()
print(w2.grad)
print(b2.grad)
print(w.grad)