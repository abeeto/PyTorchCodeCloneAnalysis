# -*- coding: utf-8 -*-
# weibifan 2022-10-1
#  多项式拟合，手工计算，不要torch

'''
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

第1步：准备训练数据。包括两部分，输入数据和预期输出。

第2步：构建模型的对象

第3步：构建损失函数

第4步：训练，设置epoch
1）前向计算。将数据传递给模型对象。
2）计算损失。
3）初始化权重为0,
4）反向传播，计算梯度，
5）更新梯度。需要设置步长，不同函数不同步长。

第5步：获得权重

'''

import numpy as np
import math
#
# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    # 链式方法，计算每个权重的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

