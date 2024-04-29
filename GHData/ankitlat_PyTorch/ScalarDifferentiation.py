#####Demonstration of PyTorch Scalar Differentiation

####Import torch module
import torch 

###data
###Let x be a variable which doesn't requires any gradient i.e it is a constant
x = torch.randn(1, requires_grad=False)

####Initialize variable for a quadratic model
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)

####Define a scalar function
y = a * x *x + b*x + c

###This will calculate gradient of y w.r.t each of the variables (a, b and c) 
y.backward()

###Print these values
print('Expected gradient of y w.r.t to a {}, Calculated Gradient {}'.format(x*x, a.grad.data))
print('Expected gradient of y w.r.t to b {}, Calculated Gradient {}'.format(x, b.grad.data))
print('Expected gradient of y w.r.t to c {}, Calculated Gradient {}'.format(1.0, c.grad.data))
