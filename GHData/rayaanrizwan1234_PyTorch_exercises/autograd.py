import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([0.1,1.0, 0.001], dtype=torch.float32)
# Z needs to be a scalar to work
z.backward(v) # will calculate the gradient of z with respect to x dz/dx
print(x.grad)

# Prevent tracking gradient while training
# x.requires_grad_(False)

# how gradients are accumulated throught the iterations which is wrong
print("Accumulated gradients")
weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    #Without this itd be wrong
    weights.grad.zero_()


