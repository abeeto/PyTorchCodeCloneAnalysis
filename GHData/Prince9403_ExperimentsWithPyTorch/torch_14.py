import torch
import torch.nn as nn


a = torch.tensor(10.0, requires_grad=True)
optimizer = torch.optim.Adam([a], lr=0.01)

print(f"Initially a={a}, torch.abs(a-7.0)={torch.abs(a-7.0)}")

for i in range(100):
    b = a - 7.0
    loss = nn.functional.relu(b)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"After optimization a={a}")


alpha = torch.tensor(5.0, requires_grad=True)
print(f"Initially alpha={alpha}, alpha.grad={alpha.grad}")
x = torch.zeros(2)
x[0] = alpha ** 2
x[1] = alpha ** 3
print(f"x={x}, x.mean={x.mean()}")
z = x[0] + x[1]
z -= 4
print(f"z={z}. Does z require grad? {z.requires_grad}")
# z -= z/2.0
z.backward()
print(f"After backprop alpha.grad={alpha.grad}")

# Error
u = torch.tensor([alpha ** 2, alpha ** 3])
v = u[0] + u[1]
# v.backward()

p = torch.log(7.0/alpha)
print(f"p={p}. type(p)={type(p)} Does p require grad? {p.requires_grad}")


