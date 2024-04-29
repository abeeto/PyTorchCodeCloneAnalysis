import numpy as np
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 3213

n_epochs = 142
alpha = 0.1

def forward(x):
    return (w * x)

def loss(x, y):
    y_pred = forward(x)
    ret = (y_pred - y) * (y_pred - y)
    return ret

def gradient(x, y):
    return (2 * x * (x * w - y))

def backward(x, y):
    curW = w
    curW = curW - alpha * gradient(x, y)
    return curW

for i in range(n_epochs):

    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_sum  += loss(x_val, y_val)
        w = backward(x_val, y_val)

    print("Progress after {} epochs => Loss : {}, W : {}".format(i, loss_sum, w))
