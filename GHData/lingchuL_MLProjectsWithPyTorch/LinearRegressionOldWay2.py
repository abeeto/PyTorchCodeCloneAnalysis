# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

X = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])

Y = torch.tensor([[2.0],
                  [4.0],
                  [6.0]])

i_historyList = []
J_historyList = []

# w = torch.tensor([1.0])
w = torch.randn(1)
w.requires_grad = True

b = torch.randn(1, requires_grad=True)


def forward(x):
	return x * w + b


def getLoss(x, y):
	y_pred = forward(x)
	m = y.size()[0]
	J = 0.5 * (1 / m) * torch.sum((y_pred - y) ** 2)
	return J


'''
print(w)
print(b)
print(forward(4))
'''

for epoch in range(1000):
	loss = getLoss(X, Y)

	i_historyList.append(epoch)
	J_historyList.append(loss.detach().numpy())

	loss.backward()
	with torch.no_grad():
		torch.sub(w, w.grad, alpha=0.1, out=w)
		torch.sub(b, b.grad, alpha=0.1, out=b)
		w.grad.zero_()
		b.grad.zero_()

print(w)
print(b)
print(forward(4))

plt.figure(1)
plt.plot(i_historyList, J_historyList)
plt.show()
