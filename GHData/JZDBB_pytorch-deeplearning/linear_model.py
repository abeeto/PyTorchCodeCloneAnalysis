import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [3, 6, 9]


w_list = []
MSE_list = [] #mean square error
y_pred = []

for w in np.arange(0.0, 5.0, 0.1):
    print(w)
    mse = 0
    for x in x_data:
        for y in y_data:
            y_pred = w * x
            mse += (y_pred-y) * (y_pred-y)
    print(mse)
    w_list.append(w)
    MSE_list.append(mse/len(x_data))

plt.plot(w_list, MSE_list)
plt.xlabel("w")
plt.ylabel("MSE")
plt.show()