import torch
import torch.nn as nn
import numpy as np


# simple model
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# instantiate model
model = Model(1, 1)

# print initial parameters
print(model.linear.weight, model.linear.bias)
for name, param in model.named_parameters():
    print(name, "\t", param.item())

# check forward pass
# x2 = torch.tensor([2.0])
# print(model.forward(x2))

# set loss function
criterion = nn.MSELoss()

# set optimisation
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.linspace(start=1, end=10000, steps=10000, dtype=torch.float32)
x = x.reshape(-1, 1)
er = torch.randn(size=(10000, 1), dtype=torch.float32)
y = 2 * x + 1  # + er / 10
print(x)
print(y)
# training
EPOCHS = 5000
losses = []
for i in range(EPOCHS):
    # forward
    y_pred = model.forward(x)
    # forward loss
    loss = criterion(y_pred, y)
    losses.append(loss)
    print(
        "epoch {} loss: {} model weight: {} model bias: {}".format(
            i, loss.item(), model.linear.weight.item(), model.linear.bias.item()
        )
    )
    optimiser.zero_grad()  # reset gradient
    # backward
    loss.backward()
    optimiser.step()  # update parameters
