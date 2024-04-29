import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import torch 
import torch.nn as nn
import torch.nn.functional as F


X, y = load_boston(return_X_y=True)
X.shape
y.shape


scaler = MinMaxScaler()


X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

X_train = torch.from_numpy(X_train.astype('float32'))
X_test = torch.from_numpy(X_test.astype('float32'))
y_train = torch.from_numpy(y_train.reshape(-1, 1).astype('float32'))
y_test = torch.from_numpy(y_test.reshape(-1, 1).astype('float32'))

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 12)
        self.layer4 = nn.Linear(12, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.layer4(x)
        return x

model = NeuralNetwork()

loss_obj = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(100):
    optimizer.zero_grad()
    for X, y in dataset_train:
        y_pred = model(X)
        loss = loss_obj(y_pred, y)
        loss.backward()
        optimizer.step()

y_pred = model(X_test)
r2_score(y_test.detach().numpy(), y_pred.detach().numpy())

