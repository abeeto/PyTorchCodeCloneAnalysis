import torch
from torch import FloatTensor
from torch.autograd import Variable
import torchvision
import numpy as np


# using forward pass
def forward(x):
  return x*w
def loss(x,y):
  y_pred = forward(x)
  return (y_pred - y) * (y_pred - y)

x_data = [11.0, 22.0, 33.0]
y_data = [21.0, 14.0, 64.0]
w = Variable(torch.Tensor([1.0]), requires_grad = True)
# before training
print("predict (before training)", 4, forward(4).data[0])
# run the training loop
for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    l = loss(x_val, y_val)
    l.backward()
    print("\tgrad: ", x_val, y_val, w.grad.data[0])
    w.data = w.data - 0.01 * w.grad.data

    # manually set the gradients to zero after updating weights
    w.grad.data.zero_()
  print("progress:", epoch, l.data[0])
# after training
print("predict (after training)", 4, forward(4).data[0])

a = Variable(FloatTensor([5]))
weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (12, 53, 91,73)]
w1, w2, w3, w4 = weights
b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
Loss = (10 - d)
Loss.backward()
for index, weight in enumerate(weights, start=1):
  gradient, *_ = weight.grad.data
  print(f"Gradient of w{index} w.r.t. Loss: {gradient}")