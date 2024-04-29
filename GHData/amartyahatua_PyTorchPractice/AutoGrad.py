import  torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
z = z.mean()

print(z)
z.backward()
print(x.grad)

# If the output is not a scalar value
print("If the output is not a scalar value")
x = torch.randn(3, requires_grad=True)
y = x+2
z = y*y*2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)

# Not gradient
# x.requires_grad(False)
# x.detach()
# with torch.no_grad()
with torch.no_grad():
    y = x+2
    print(y)



weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
weights.grad.zero_()
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD(weights, lr=0.001)
optimizer.step()
optimizer.zero_grad()