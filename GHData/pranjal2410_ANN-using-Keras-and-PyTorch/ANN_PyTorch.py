# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorizing the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN using PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

y_train = torch.squeeze(torch.from_numpy(y_train).float())
y_test = torch.squeeze(torch.from_numpy(y_test).float())


class Classifier(nn.Module):

    def __init__(self, n_features):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(n_features, 6)
        self.linear2 = nn.Linear(6, 6)
        self.linear3 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return torch.sigmoid(self.linear3(x))


classifier = Classifier(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)
device = torch.device('cuda:0')

X_train = X_train.to(device)
X_test = X_test.to(device)

y_train = y_train.to(device)
y_test = y_test.to(device)

classifier = classifier.to(device)
criterion = criterion.to(device)


def cal_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


for epoch in range(100):

    y_pred = classifier(X_train)

    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)

    train_accuracy = cal_accuracy(y_train, y_pred)
    y_test_pred = classifier(X_test)
    y_test_pred = torch.squeeze(y_test_pred)

    test_loss = criterion(y_test_pred, y_test)

    test_acc = cal_accuracy(y_test, y_test_pred)
    print(
        f'''epoch {epoch}
    Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_accuracy)}
    Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
    ''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
