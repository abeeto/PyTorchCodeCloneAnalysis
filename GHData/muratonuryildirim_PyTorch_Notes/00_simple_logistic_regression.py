import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
0. prepare dataset
1. design model (input, output, forward pass)
2. initiate the model
3. define loss and optimizer
4. train the model (loss)
    - forward pass: compute prediction and loss
    - backward pass: update weights
5. test the model (accuracy)
'''

# 0) prepare dataset
xy = datasets.load_breast_cancer()
X, y = xy.data, xy.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. design model
class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# 2. initiate the model
model = LogisticRegression(n_features)

# 3. define loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 4. train the model (loss)
num_epochs = 50
for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')

# 5. test the model (accuracy)
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = y_pred.round()
    acc = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc}')





