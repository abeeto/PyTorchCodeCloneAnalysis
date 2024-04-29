from pickletools import optimize
import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x+2
# print(y)

# z = y*y*2
# z = z.mean()
# print(z)

# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

# z.backward(v) # dz/dx
# print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)

# x.requires_grad_(False)
# x.detach() # with torch.no_grad():
 

# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

# with torch.no_grad():
#     y = x+2
#     print(y)

weights = torch.ones(4, requires_grad=True)

# for epoch in range(3):
#     model_output = (weights*3).sum()

#     model_output.backward() 

#     print(weights.grad)

#     weights.grad.zero_()

# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()

