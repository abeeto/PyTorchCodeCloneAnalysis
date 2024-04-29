# Uses a fully connceted LR NN model
import torch
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn

df = pd.read_csv("bike-sharing-data.csv", index_col=0)
print(df.columns)

# preprocess categorical column
data = pd.get_dummies(df, columns=["season"])

print(data.sample(5))
print(data.columns)

columns = ['holiday', 'workingday', 'weather', 'temp', 'atemp','registered', "season_1", 'season_2',
           'season_3', 'season_4', "registered"]

features = data[columns]
target = data[["count"]]

X_train, x_test, Y_train, y_test = train_test_split(
                                                        features,
                                                        target,
                                                        test_size=0.2
                                                    )

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float)

x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

import torch.utils.data as data_utils
train_data = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = data_utils.DataLoader(train_data, batch_size=400, shuffle=True)

print(len(train_loader))