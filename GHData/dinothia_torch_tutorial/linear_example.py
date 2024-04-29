# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import math

swap_in_out = False
# Create training data
N = 10

A = torch.randn(1, 1)

D_I = A.shape[0]
D_O = A.shape[1]

b = 0
#x = torch.randn(N, D_I)
x = torch.linspace(-5, 5, N).view(N, 1)
y = x @ A + b
#y += 0.01 * torch.randn_like(y)

if swap_in_out:
    x_train = y
    y_train = x

    D_I = A.shape[1]
    D_O = A.shape[0]
else:
    x_train = x
    y_train = y

# Add to dataloader
bs = 32
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs)

#lr = 1e-3 #* (10**3 / 10**D_I)i
lr = 1e-3 
model = torch.nn.Sequential(
    torch.nn.Linear(D_I, D_O),
    #torch.nn.ReLU()
)

loss_fn = torch.nn.MSELoss(reduction="mean")
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_arr = []
epochs = 100
for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    #if epoch % 10 == 9:
    loss_arr.append(loss.item())
    print(f"epoch: {epoch} loss: {loss.item()}")

linear_layer = model[0]

if swap_in_out:
    print("xy swapped:")
print(f"bias term = \n{linear_layer.bias.detach().numpy()}")
print(f"actual weight = \n {A.detach().numpy()}")
print(f"inv actual weight = \n {np.linalg.inv(A.detach().numpy())}")
print(f"weights term = \n{linear_layer.weight.detach().numpy().round(2).T}")
# print(x_train[-1])

plt.plot(loss_arr)
plt.show()
if swap_in_out:
    plt.imshow(np.linalg.inv(A.detach().numpy()))
    plt.show()
plt.show()

print(f"inv actual weight row 0 = \n {np.linalg.inv(A.detach().numpy())[0]}")
print(f"weights term row 0 = \n{linear_layer.weight.detach().numpy().T[0]}")
