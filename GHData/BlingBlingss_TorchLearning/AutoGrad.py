# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
import torch
from torch.autograd import Variable

#  If you set its attribute .requires_grad as True, it starts to track all operations on it.
x = torch.ones(2, 2, requires_grad=True)
print("x:", x)
y = x + 2
# y was created as a result of an operation, so it has a grad_fn.
print("y:", y)
z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
# 同a.requires_grad = True
print(a.requires_grad)
b = (a * a).sum()
print(b)
print("b,grad_fn:", b.grad_fn)

# Let’s backprop now. Because out contains a single scalar(out=1/4 * 3 * (x+2)^2), out.backward() is equivalent to out.backward(torch.tensor(1.)).
out.backward()
# Print gradients d(out)/dx
print("x.grad:", x.grad)

# Now let’s take a look at an example of vector-Jacobian product:
x = torch.ones(3, requires_grad=True)
y = x * 2
print("y.data:", y.data)
while y.data.norm() < 1000:
    # torch.norm()是对输入的Tensor求范数,默认二范数
    print("y.data.norm():", y.data.norm())
    y = y * 2
print(y)
# v只是对雅可比矩阵（y对x求导得出的雅可比矩阵）进行线性操作的变量
# # backward（）的参数[k1,k2,k3....kn]的含义就是：https://www.cnblogs.com/JeasonIsCoding/p/10164948.html
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
# 解释上述backward原因
a = Variable(torch.FloatTensor([[2., 4.]]), requires_grad=True)
b = torch.zeros(1, 2)
b[0, 0] = a[0, 0] ** 2 + a[0, 1]
b[0, 1] = a[0, 1] ** 3 + a[0, 0]
out = 2 * b
# backward（）的参数[k1,k2,k3....kn]的含义就是：https://www.cnblogs.com/JeasonIsCoding/p/10164948.html
out.backward(torch.FloatTensor([[1., 1.]]))
print("a:", a.data)
print("out:", out.data)
print(a.grad)

print(x.requires_grad)