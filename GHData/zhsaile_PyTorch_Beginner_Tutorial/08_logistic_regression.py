import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

#print(y)
n_samples, n_features = X.shape
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, num_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_input_features, 1)

    def forward(self, x):
        output = self.linear(x)
        y_predicted = torch.sigmoid(output)
        return y_predicted

model = LogisticRegression(n_features)
# 2) loss and optimizer
lr = .01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3) training loop
epochs = 1000
for epoch in range(epochs):
    # forward pass and loss calculate
    y_predicted = model(X_train)
    l = loss(y_predicted, y_train)
    # backward
    l.backward()
    # updates
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
