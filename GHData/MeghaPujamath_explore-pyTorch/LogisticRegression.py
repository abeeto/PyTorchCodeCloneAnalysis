# 1. Design model
# 2. Calculate Loss and Optimizer
# 3. Training Loop
#       a. forward pass: compute prediction
#       b. backward pass: gradients
#       c. update weights

import imp
from statistics import mode
from numpy import dtype
from sympy import N
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()

        # define layers
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# A. Data prepration


# 1. Define train and test data
data = datasets.load_breast_cancer()
X, y = data.data, data.target

n_samp, n_featu = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# 2. Scaling the input features
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# 3. reshape y
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# B. Define Model
model = LogisticRegression(n_featu)

# C. calculate loss = Binary cross entropy loss
learning_rate = 0.01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# D. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # 1. forward pass and loss calculation
    y_pred = model.forward(X_train)

    l = loss(y_pred, y_train)

    # 2. gradient calculation - backward pass
    l.backward()

    # 3. update weights
    optimizer.step()

    # 4. empty gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        #[w, b] = model.parameters()
        print(f"epoch {epoch+1}: loss = {l.item():.4f}")

# Evaluation
with torch.no_grad():
    y_pred = model.forward(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy : {acc:.4f}')
