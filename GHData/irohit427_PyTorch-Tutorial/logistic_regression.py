import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
input_dim = X_train.shape[1]

class LogisticRegression(nn.Module):
  
  def __init__(self, input_dim, output):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_dim, output)
  
  def forward(self, x):
    return torch.sigmoid(self.linear(x))

model = LogisticRegression(input_dim, 1)
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
iterations = 100

for epoch in range(iterations):
  y_pred = model(X_train)
  
  loss = loss_func(y_pred, y_train)
  
  loss.backward()
  
  optimizer.step()
  
  optimizer.zero_grad()
  if (epoch + 1) % 10 == 0:
    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    
with torch.no_grad():
  predicted = model(X_test)
  predicted = predicted.round()
  
  acc = predicted.eq(y_test).sum() / float(y_test.shape[0])
  print(f'accuracy = {acc:.4f}')