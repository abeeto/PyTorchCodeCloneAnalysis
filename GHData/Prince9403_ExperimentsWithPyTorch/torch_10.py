import torch
import torch.nn as nn

lin_layer = nn.Linear(in_features=1, out_features=1)

x0 = 7 * torch.ones(1)
x0 = x0.view(1, 1, 1)

optimizer = torch.optim.Adam(lin_layer.parameters(), lr=0.01)

print("x0 size:", x0.size())

for i in range(10):
    optimizer.zero_grad()
    out = lin_layer(x0)
    out.backward()
    optimizer.step()
print("out=", out)