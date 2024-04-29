# 1 design model (input, otput, forwardpass)
# 2 Construct the loss and optimizer
# 3 Training loop
# - forward pass: compute prediction
# - backprop: gradients
# - update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0 prepare data (generate random data)
X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features=1, noise = 20, random_state=1)

# rn X_numpy is a double
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshapes y from a column vec to a row vec
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

inputSize = n_features
outSize = 1

# 1 Model
model = nn.Linear(inputSize, outSize)
lr = 0.01
n_iter = 100

# 2 loss and optim
criterion = nn.MSELoss() # Loss
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3 training loop
for epoch in range(n_iter):
    # Forward pass
    yPred = model(X)

    #loss
    l = criterion(yPred, y)

    # Backprop
    l.backward()

    # weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss {l.item():.4f}')

#plot (prevent from being tracked)
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
