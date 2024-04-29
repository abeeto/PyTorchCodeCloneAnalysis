import torch

# autograd
x = torch.ones(2, 2, requires_grad=True)
print(x, x.grad_fn)

y = x + 2
print(y, y.grad_fn)
print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, z.grad_fn)
print(out, out.grad_fn)

print('\n', 'inplace改变requires_grad的状态')
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)  # inplace change state
print(a.requires_grad)
b = (a * a).sum()
print(b, b.grad_fn)  # 为什么sumbsck0和之前z的mulback0的地址一样？？
print(torch.tensor(1.))

print('\n', 'grad 累加的性质')
out.backward()
print(x.grad)
# print(y.grad)
#  If you indeed want the .grad field to be populated for a non-leaf Tensor,
#  use .retain_grad() on the non-leaf Tensor.
out2 = x.sum()
print(out2, out2.grad_fn)
out2.backward()  # 这里相当于另一个树枝，前一个走到out，这一个单独走到out2
# x.grad是这两个分支单数的相加
print(x.grad)

out3 = x.sum() * 3  # 第三个分支
print(out3, out3.grad_fn)
x.grad.data.zero_()  # 所有关于x.grad的数据清零
out3.backward()
print(x.grad)

print('\n', '张量backward的情况处理：创建同形weight张量，传给backward函数')
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
v = torch.tensor([[1, 0.1], [0.01, 0.001]], dtype=torch.float)
# z.backward(v)  # 等价于后面两句程序
h = torch.sum(z * v)
h.backward()
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
print(x.data)  # 还是一个tensor
# x.data.requires_grad = True  # 毫无效果
print(x.data.requires_grad)  # 但是已经是独立于计算图之外，why？？？
y = 2 * x ** 2 + 2  # grad:4*x,包含x，此时，按照x=100计算
# Y = 2 * x * x #等价于y1=2*x，y=y1*x。---grad：2+y1=2+2*x
# y = 2 * x
x.data *= 100
y.backward()
print(x)
print(x.grad)
