"""
PYTORCH PIPELINE

1 -> Design Model
2 -> Construct Loss and Optimizer
3 -> Training Loops
        -> forward pass
        -> backward pass
        -> update weights
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import style
# style.use('fivethirtyeight')
import copy

# data preperation

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(-1,1)
y_test = y_test.view(-1,1)
# Model Architecuture
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features, output_size = 1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, output_size)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_input_features=n_features)

# Loss and Optimizer
loss_func = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Process
for epoch in range(100):
    
    # FORWARD
    y_predicted = model(X_train)
    # y_forplot = model(X)
    loss = loss_func(y_predicted, y_train)

    # BACKWARD
    loss.backward() #backpropogation

    # UPDATE
    optimizer.step()
    optimizer.zero_grad()
 
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc:.4f}')