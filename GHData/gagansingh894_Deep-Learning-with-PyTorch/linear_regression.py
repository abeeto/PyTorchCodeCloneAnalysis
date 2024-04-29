"""
PYTORCH PIPELINE

1 -> Design Model
2 -> Construct Loss and Optimizer
3 -> Training Loops
        -> forward pass
        -> backward pass
        -> update weights
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import style
# style.use('fivethirtyeight')
import copy

# data preperation

X_numpy, y_numpy = datasets.make_regression(n_samples=100000, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32)).view(y_numpy.shape[0],1)

n_samples, n_features = X.shape

# Model Architecuture
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Process
num_epochs = 100
fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

def animate(num_epochs=100):


    for epoch in range(num_epochs):
        
        # FORWARD
        y_predicted = model(X)
        y_forplot = model(X)
        loss = loss_func(y_predicted, y)

        # BACKWARD
        loss.backward() #backpropogation

        # UPDATE
        optimizer.step()
        optimizer.zero_grad()
        plt.cla()
        plt.scatter(X_numpy, y_numpy)
        plt.plot(X_numpy, y_forplot.detach().numpy(), 'g')    

        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')


# predicted = model(X).detach().numpy()
# sns.scatterplot(X_numpy.flatten(), y_numpy)
# # plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'g')
# plt.show()
ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()

