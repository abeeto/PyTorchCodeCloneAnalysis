import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

os.system('cls')

# 0) generate data:
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
n_samples, n_features = x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)

#scale:
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model:
class LogisticRegression(nn.Module):

    def __init__(self, inp_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(inp_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

inp_size = n_features
out_size = 1
model = LogisticRegression(inp_size)

# 2) loss and opt:
learning_rate = 0.01
criterion = nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training stage:
n_epochs = 200
for epoch in range(n_epochs):

    #forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    #backward pass
    loss.backward()

    #update weights
    opt.step()
    opt.zero_grad()
    if epoch%10 == 0:
        print(f'epoch : {epoch}, loss : {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(x_test).round()
    acc = y_pred.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy : {acc*100:.3f}')