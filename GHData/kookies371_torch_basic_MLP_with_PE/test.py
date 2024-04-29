from importlib.metadata import requires
import torch
import torch.nn as nn

a = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2, 2), requires_grad=True)

optimizer = torch.optim.Adam([b], lr=1e-3)
loss_fn = nn.MSELoss()

print("before train")
print(b)

loss = loss_fn(a, a+b)
loss.backward()
optimizer.step()

print("after train")
print(b)


class model_base:
    def __init__(self, n, n_units) -> None: # u_units: [784, 20, 20, 10]
        moudelist = nn.ModuleList()
        for i in range(n):
            modulelist.append(nn.Linear(n_units, ))

new_model = model_base()
new_model.initialize(model.encoder[:-1], strict=False)
