# Linear_Regression

import torch 
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data 

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=1)


X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape 
#  model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


# loss and optimizer 
lr = 0.01
criterion  = nn.MSELoss() #  callabe function

optimizer  = torch.optim.SGD(model.parameters(), lr=lr)

# traning loop 

n_epoch = 100

for epoch in range(n_epoch):

    # fwd pass

    y_predctited = model(X)
    loss = criterion(y_predctited, y)

    #  backward pass

    loss.backward()


    #  update

    optimizer.step()


    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:

        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')
predctied = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'r')
plt.plot(X_numpy, predctied, 'b')