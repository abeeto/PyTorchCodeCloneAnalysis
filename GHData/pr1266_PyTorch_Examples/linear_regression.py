import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import os

os.system('cls')

# 0) generate data:
X_numpy, Y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model:
inp_size = n_features
out_size = 1
model = nn.Linear(inp_size, out_size)

# 2) loss and opt:
learning_rate = 0.01
criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training stage:
n_epochs = 200
for epoch in range(n_epochs):

    #forward pass
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    #backward pass
    loss.backward()

    #update weights
    opt.step()
    opt.zero_grad()
    if epoch%10 == 0:
        print(f'epoch : {epoch}, loss : {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

    