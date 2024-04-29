# -*- coding:utf-8 -*-

import math
import numpy as np
import torch

X = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])

Y = torch.tensor([[2.0],
                  [4.0],
                  [6.0]])

w = torch.randn(1)
w.requires_grad = True

b = torch.randn(1, requires_grad=True)

ComputeDevice = "cuda" if torch.cuda.is_available() else "cpu"


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x * w + b


def getLoss(y_pred, y):
    m = y.size()[0]
    J = 0.5 * (1 / m) * torch.sum((y_pred - y) ** 2)
    return J


def train(model):
    for epoch in range(1000):
        pred = model(X)
        loss = getLoss(pred, Y)
        loss.backward()
        with torch.no_grad():
            torch.sub(w, w.grad, alpha=0.01, out=w)
            torch.sub(b, b.grad, alpha=0.01, out=b)
            w.grad.zero_()
            b.grad.zero_()
    print(w)
    print(b)
    print(model(4))


Model = LinearRegressionModel()
Model.to(device=ComputeDevice)
train(Model)
