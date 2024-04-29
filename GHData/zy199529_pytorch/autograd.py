import torch as t
from torch.autograd import Variable

x = Variable(t.ones(2, 2), requires_grad=True)
print(x)
y = x.sum()
y.backward()
y.backward()
print(x.grad)