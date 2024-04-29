# 1. Design model
# 2. Calculate Loss and Optimizer
# 3. Training Loop
#       a. forward pass: compute prediction
#       b. backward pass: gradients
#       c. update weights

from numpy import dtype
from sympy import N
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()

        # define layers
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape y
y = y.view(y.shape[0], 1)

# 1. define linear model
n_samp, n_featu = X.shape
input_size = n_featu
output_size = 1

model = LinearRegression(input_size, output_size)

# 2. calculate loss = Mean Square Error
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # 1. forward pass and loss calculation
    y_pred = model.forward(X)

    l = loss(y, y_pred)

    # 2. gradient calculation - backward pass
    l.backward()

    # 3. update weights
    optimizer.step()

    # 4. empty gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

# plot
predicted = model.forward(X).detach()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
