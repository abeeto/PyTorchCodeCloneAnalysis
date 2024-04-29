# Linear Relationship y = x * weight<this will be random> + bias
# Training Loss (error): loss = (y(hat) - y) ^ 2    // hat = predicted y

import numpy as np
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [2,4,6]

# forward linear function [ y = x * w ] bias is removed
def forward(x):
    return x * w

# loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

w_list = []
mse_list = []

# computing loss for each w
for w in np.arange(0,4.1,0.1):
    print("w = ",w)
    l_sum = 0

    for each_x,each_y in zip(x_data,y_data):
        each_pred_y = forward(each_x)
        l = loss(each_x,each_y)
        l_sum += l
        print("\t", each_x,each_y,each_pred_y,l)

    # MSE
    print("MSE = ", l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum/ 3)


#Plot graph
plt.plot(w_list,mse_list)
plt.xlabel("w")
plt.ylabel("Loss")
plt.show()
