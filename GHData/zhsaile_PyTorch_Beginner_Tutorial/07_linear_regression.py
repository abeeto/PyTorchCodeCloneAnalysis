import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1,
         noise=20, random_state=1)
y_numpy = y_numpy.reshape((100,1))


X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

#plt.scatter(X, y)
#plt.show()

#print(X_numpy)
#print(y_numpy)

# 1) define model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
# 2) loss and optimizer
lr = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3) training loop
epochs = 1000

for epoch in range(epochs):
    #forward
    output = model(X)

    #calculate loss
    l = loss(output, y)

    #backward & update
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 100 ==0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

