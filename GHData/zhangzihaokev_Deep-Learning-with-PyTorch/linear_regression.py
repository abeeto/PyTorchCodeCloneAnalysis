# pytorch training pipeline:
# 1. design model (input, output sizes, forward pass)
# 2. construct loss and optimizer
# 3. training loop
#       - forward pass: compute prediction
#       - backward pass: compute gradients
#       - update weights

from scipy.fft import next_fast_len
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# design model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# define loss and optimizer
learning_rate = 0.01
criterion  = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass, backward() will sum the gradients into .grad attribute of optimizer so we need to zero it in between runs
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}, loss = {loss.item():.4f}')

# plot, use detach to generate new tensor without requires_grad=True
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'bs')
plt.show()

