"""
Description: We will try to implement linear regression using pytorch
Steps: 
    0. Prepare Data
    1. Design Model, choose Loss function, choose optimizer
    2. Create a Training Loop
        a. Forward pass
        b. Calculate loss
        c. Backward pass
        d. Take optimizer step
        e. re-iterate for n_epochs

"""
import torch
import torch.nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Step 0: Prepare Data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1337)

# Model input should be a tensor, not numpy array. so we need convert this data to tensors
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

# Step 1: Design Model
in_samples, in_features = X.shape
out_features = 1
model = torch.nn.Linear(in_features, out_features)

# Define Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Step 2: Construct a training loop
for epoch in range(100):
    # Forward Pass
    y_hat = model(X)

    # Loss
    loss = criterion(y_hat, y)

    # Backward pass
    loss.backward()

    # optimizer step
    optimizer.step()

    # set grads to zero
    optimizer.zero_grad ()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")

y_preds = model.forward(X).detach().numpy()

# Plot the results
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, y_preds, 'b')
plt.show()  

    