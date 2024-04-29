import torch

# 自动求梯度
x = torch.ones(2, 2, requires_grad=True)
print(x, x.grad_fn, x.is_leaf)
y = x + 2
print(y, y.grad_fn, y.is_leaf)
z = y * y * 3
out = z.mean()
print(z, z.grad_fn)
print(out, out.grad_fn)

print('\n', 'inplace改变requires_grad的状态')
a = torch.ones(2, 2)
a = a * 3
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad, a.grad_fn)
b = a + 2
print(b, b.grad_fn)  # 为什么b.grad_fn和之前z的out.grad_fn的地址一样？？

print('\n', 'grad 累加的性质')
out.backward()
print(x.grad)
out2 = x.sum()
print(out2, out2.grad_fn)
out2.backward()  # 这里相当于另一个树枝，前一个走到out，这一个单独走到out2
# 这次回退，只计算out2的grad，理论上讲应该为1，由于上面out回退时，x.grad有值，故累计
print(x.grad)
out3 = x.sum() * 3
x.grad.data.zero_()  # 将x.grad的历史数据清零
out3.backward()  # 回退，计算当前支的grad
print(x.grad)

print('\n', '张量backward的情况处理：创建同形weight张量，传给backward函数')
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = x * 2
z = y.view(2, 2)
print(z)
v = torch.tensor([[1, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
# 上面一句等价于后面两句程序
# h = torch.sum(z * v)
# h.backward()
print(x.grad)

print('\n', '中断追踪的例子')
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)
y3.backward()
print(x.grad)

print('\n', '只改变tensor的值，对grad没有影响')
x = torch.ones(1, requires_grad=True)
print(x.data)
# x.data.requires_grad
print(x.data.requires_grad) # 这里书上是不是讲错了啊，完全没起到效果
# y = 2 * x ** 2 + 2
y = 2 * x
x.data *= 100
y.backward()
print(x)
print(x.grad)
