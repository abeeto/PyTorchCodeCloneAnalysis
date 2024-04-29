from statistics import mode
import torch

"""
auto grad package is used  to update gradients
"""
x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y) # add_Backward_fn

z = y*y*2
print(z) # mul_backward_fn 

z1 = z.mean() # mean_backwar_fn
print(z1)

z1.backward() #dz/dx
print(x.grad)

"""
How to remove gradient function
"""
# option_1
x1 = torch.rand(3, requires_grad=True)
print(x)
x1.requires_grad_(False)
print(x1)

# option_2
x2 = torch.rand(5, requires_grad=True)
y2 = x2.detach()
print(y2)

# option_3
x3 = torch.rand(7, requires_grad=True)
with torch.no_grad():
    y = x + 3
    print(y)


### dummy traininig example

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_() #empty the weights
