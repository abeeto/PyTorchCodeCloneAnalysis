# 这是一个使用pytorch的简单实例
import torch
from torch.autograd import Variable

N, D = 100, 50
learning_rate = 10
# 将其转化为变量
x = Variable(torch.randn(N, D), requires_grad=True)
y = Variable(torch.randn(N, D), requires_grad=True)
z = Variable(torch.randn(N, D), requires_grad=True)

for i in range(5):
    print("第{}次循环".format(i))
    a = x + y
    b = a * z
    c = torch.sum(b)
    c.backward()
    # print(x.grad.data)
    # print(y.grad.data)
    # print(z.grad.data)
    x.data += learning_rate * x.grad.data
    y.data += learning_rate * y.grad.data
    z.grad += learning_rate * z.grad.data
    print("loss:{}".format(c))
