# 1) Prepare Data
# 2) Design model (input, output size, forward pass)
# 3) Construct loss and optimizer
# 4) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights
#   - empty gradients
# 5) Training loop

import torch
import torch.nn as nn
import numpy as np

# import dataset and matplot for visualization
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Steps
# *************************************************************************************************
# 0) Prepare data
# *************************************************************************************************
bc = datasets.load_breast_cancer()
X, y = bc["data"], bc["target"]

n_samples, n_features = X.shape
# print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# cast train and test data to torch floats
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# cast target to column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Design model
# f = wx + b, sigmoid at the end
# *************************************************************************************************
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(n_features)


# 2) Construct Loss and Optimizer
# *************************************************************************************************
# a. Loss --> Binary Cross Entropy Loss
learning_rate = 0.01
criterion = nn.BCELoss()
# b. Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3) Training
# *************************************************************************************************
num_epochs = 100

for epoch in range(num_epochs):

    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss: {loss.item():.4f}")


# 4) Evaluate model
# ***********************************************************************************
# should not be part of computational graph cos we do not want to track the history
with torch.no_grad():
    y_pred = model(X_test)
    # the immediate output of the model will be confidence scores
    # to cast them to classes, i.e 0 or 1, we use the round() method
    y_pred_cls = y_pred.round()
    # accuracy ---> sum number of observations where predicted labels is equal to
    # actual labels and divide by the number of observations of test data
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy:{acc:0.4f}")
