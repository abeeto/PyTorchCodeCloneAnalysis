# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np

# import regression dataset and matplot for visualization
from sklearn import datasets
from matplotlib import pyplot as plt

# Steps
# *************************************************************************************************
# 0) Prepare data
# *************************************************************************************************
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)
# print(f"y_numpy: {y_numpy}")

# cast np.arrays to float tensors
X = torch.from_numpy(X_numpy.astype(np.float32))
# make sure that target is a column vector not a flat vector
y = torch.from_numpy(y_numpy.astype(np.float32))
# in PyTorch the equivalent of the reshape method is view
y = y.view(y.shape[0], 1)
# print(f"y:{y}")

# define input and output size
n_samples, n_features = X.shape


# 2) Design model
# *************************************************************************************************
# the number of features
input_size = n_features
# the number of values for each sample
output_size = 1

model = nn.Linear(input_size, output_size)

# 3) Construct Loss and Optimizer
# *************************************************************************************************
# a. Loss
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 4) Training
# *************************************************************************************************
num_epochs = 100

for epoch in range(num_epochs):

    # forward pass: compute prediction and loss
    y_pred = model(X)
    print(f"y_pred:{y_pred[0][0]}")
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    # empty gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss= {loss.item():0.4f}")

# plot
# by default,the model sets the requires_grad to be true and to visualize it we must not track the gradient 
# in the computational graph
# convert predicted labels to numpy array
predicted = model(X).detach().numpy()

print(predicted)
print(y_numpy)
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show

# Debug why predicted seems so different from y
