# -*- coding:utf-8 -*-

import math
import numpy as np
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w = torch.tensor([1.0])
w = torch.randn(1)
w.requires_grad = True

b = torch.randn(1, requires_grad=True)


def forward(x):
	return x * w + b


def loss(x, y):
	y_pred = forward(x)
	return (y_pred - y) ** 2


print(w)
print(forward(4))

l = 0

for epoch in range(50):
	for x, y in zip(x_data, y_data):
		l = loss(x, y)
		l.backward()
		w.data = w.data - 0.01 * w.grad.data
		b.data = b.data - 0.01 * b.grad.data
		w.grad.data.zero_()
		b.grad.data.zero_()

print(w)
print(b)
print(forward(4))
