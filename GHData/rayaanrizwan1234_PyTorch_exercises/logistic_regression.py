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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0 prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Train test split is a model validation procedure that allows you to simulate how a model would perform on new/unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale make our data have 0 mean
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Make it a row vecor
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1 Model
# f = wx + b, sigmoid at the end
class logisticRegression(nn.Module):
    def __init__(self, nInputFeatures):
        super(logisticRegression, self).__init__()
        self.lin = nn.Linear(nInputFeatures, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred

# Create our model
model = logisticRegression(n_features)

# loss compute the binary cross entropy loss between probabilities
lr = 0.01
criterion = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop

niter = 200

for epoch in range(niter):
    # Forward and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()
    # update
    optimiser.step()
    # empty gradients
    optimiser.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    # Rounds to 0 or 1
    yPredClasses = y_pred.round()
    # If ypredclasses is eq to y_test then it adds 1
    acc = yPredClasses.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.3f}')




