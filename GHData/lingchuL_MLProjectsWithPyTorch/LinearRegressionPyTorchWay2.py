# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

ComputeDevice = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.tensor([[1.0],
                  [2.0],
                  [3.0]], device=ComputeDevice)

Y = torch.tensor([[2.0],
                  [4.0],
                  [6.0]], device=ComputeDevice)

i_historyList = []
J_historyList = []


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.LayerFunc = torch.nn.Linear(1, 1, device=ComputeDevice)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.LayerFunc(x)


def getLoss(y_pred, y):
    LossFunc = torch.nn.MSELoss()
    J = 0.5 * LossFunc(y_pred, y)
    return J


def train(model, optimizer, iternum):
    for epoch in range(iternum):
        optimizer.zero_grad()

        pred = model(X)

        loss = getLoss(pred, Y)
        loss.backward()

        optimizer.step()

        J_historyList.append(loss.detach().item())
        i_historyList.append(epoch)

    print(model(torch.tensor([[4]], dtype=torch.float, device=ComputeDevice)))


Model = LinearRegressionModel()
Model.to(device=ComputeDevice)

Optimizer = torch.optim.SGD(Model.parameters(), lr=0.35)

train(Model, Optimizer, 100)

torch.save(Model.state_dict(), r".\Model.pth")

Model2 = LinearRegressionModel()
Model2.load_state_dict(torch.load(r".\Model.pth"))

print(Model2(torch.tensor([[4]], dtype=torch.float, device=ComputeDevice)))

plt.figure(1)
plt.plot(i_historyList, J_historyList)
plt.show()
