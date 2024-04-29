import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

## Prepare Data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))
X_tensor = X_tensor.view(X.shape[0], 1)
y_tensor = y_tensor.view(y.shape[0], 1)

n_samples, n_features = X_tensor.shape
input_dim = n_features
output = 1
class LinearRegression(nn.Module):
  
  def __init__(self, input_dim, output):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(input_dim, output)
    
  def forward(self, x):
    return self.linear(x)

loss_func = nn.MSELoss()
iterations = 200
model = LinearRegression(input_dim, output)
optimizer = SGD(model.parameters(), lr= 1e-2)

for epoch in range(iterations):
  y_pred = model(X_tensor)
  loss = loss_func(y_pred, y_tensor)
  
  loss.backward()
  
  optimizer.step()
  
  optimizer.zero_grad()
  
  if (epoch + 1) % 10 == 0:
    print(f'Epoch {epoch + 1}, loss = {loss.item():.4f}')
  
predicted = model(X_tensor).detach().numpy()

plt.plot(X, y, 'ro')
plt.plot(X, predicted, 'b')
plt.show()