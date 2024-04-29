import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1


def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

w_list = []
mse_list = []

for w in np.arange(0.0, 4.0, 0.1):
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_sum += loss(x_val, y_val)
    w_list.append(w)
    mse_list.append(loss_sum/3)

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.show()