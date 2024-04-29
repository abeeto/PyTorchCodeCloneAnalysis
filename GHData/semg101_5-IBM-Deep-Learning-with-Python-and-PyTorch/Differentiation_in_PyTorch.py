# These are the libraries will be useing for this lab.

import torch 
import matplotlib.pylab as plt
import torch.functional as F

#Derivatives
#Let us create the tensor x and set the parameter requires_grad to true because you are going to take the derivative of the tensor.
# Create a tensor x
x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)

# Create a tensor y according to y = x^2
y = x ** 2
print("The result of y = x^2: ", y)

# Take the derivative. Try to print out the derivative at the value x = 2
y.backward()
print("The dervative at x = 2: ", x.grad)

# Calculate the y = x^2 + 2x + 1, then find the derivative 
x = torch.tensor(2.0, requires_grad = True)
y = x ** 2 + 2 * x + 1
print("The result of y = x^2 + 2x + 1: ", y)
y.backward()
print("The dervative at x = 2: ", x.grad)

#Practice--------------------------------------------------------------------------------------------
# Practice: Calculate the derivative of y = 2x^3 + x at x = 1
x = torch.tensor(1.0, requires_grad=True)
y = 2 * x ** 3 + x
y.backward()
print("The derivative result: ", x.grad)

#Partial Derivatives---------------------------------------------------------------------
# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2

u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)

# Calculate the derivative with respect to u
f.backward()
print("The partial derivative with respect to u: ", u.grad)

# Calculate the derivative with respect to v
print("The partial derivative with respect to u: ", v.grad)

# Calculate the derivative with multiple values
x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)

# Take the derivative with respect to multiple value. Plot out the function and its derivative
y.backward()

plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

#The relu activation function is an essential function in neural networks. We can take the derivative as follows: 
import torch.nn.functional as F

# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative
x = torch.linspace(-3, 3, 100, requires_grad = True)
Y = F.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

#Practice--------------------------------------------------------------------
# Practice: Calculate the derivative of f = u * v + (u * v) ** 2 at u = 2, v = 1
u = torch.tensor(2.0, requires_grad = True)
v = torch.tensor(1.0, requires_grad = True)
f = u * v + (u * v) ** 2
f.backward()
print("The result is ", u.grad)