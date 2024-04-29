import torch
from torch.autograd import Variable

x_tensor = torch.arange(0, 10, step=0.5)
# print(x_tensor)

# x_variable = Variable(x_tensor)
# # print(x_variable)
#
# print(x_variable.data)
# print(x_variable.grad)
# print(x_variable.requires_grad)
#
# x_variable = Variable(x_tensor, requires_grad=True)
# print(x_variable.data)
# print(x_variable.grad)
# print(x_variable.requires_grad)
#
# x_variable = Variable(x_tensor, volatile=True)
# print(x_variable.grad, x_variable.requires_grad, x_variable.volatile)

# x_variable = Variable(x_tensor, volatile=True, requires_grad=True)
# print(x_variable.grad, x_variable.requires_grad, x_variable.volatile)

x = Variable(x_tensor, requires_grad=True)
y = 2 * x
z = 0.125 * (y**2)

loss = torch.ones(20) * 0.1
z.backward(loss)

print(z.grad)
print(y.grad)
print(x.grad)

# y = x**2 + 4 * x

# z = 2 * y + 3
# print(z)
# print(x.requires_grad, y.requires_grad, z.requires_grad)
# print(y)
# print(z)

# .backward(gradient,retain_graph,create_graph,retain_variables)
# compute gradient of current variable w.r.t. graph leaves

# print(x.grad)
# z.backward(loss)
# print(x.grad)
# print(y.grad)
# print(z.grad)
# # y.grad,z.grad
