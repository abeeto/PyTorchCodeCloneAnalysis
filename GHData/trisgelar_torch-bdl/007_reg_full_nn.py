# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 19:26:56 2018

@author: trisgelar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch

automobile_data = pd.read_csv('data/Automobile_data.csv', sep=r'\s*,\s*', engine='python')

automobile_data.head()

automobile_data = automobile_data.replace('?', np.nan)
automobile_data = automobile_data.dropna()

col = ['make', 'fuel-type', 'body-style', 'horsepower']
automobile_features = automobile_data[col]
automobile_features.head()
automobile_target = automobile_data[['price']]
automobile_target.head()

automobile_features['horsepower'].describe()
pd.options.mode.chained_assignment = None

automobile_features['horsepower'] = pd.to_numeric(automobile_features['horsepower'])
automobile_features['horsepower'].describe()

automobile_target = automobile_target.astype(float)
automobile_target.describe()

automobile_features = pd.get_dummies(automobile_features, columns=['make', 'fuel-type', 'body-style'])

automobile_features.columns


automobile_features[['horsepower']] = preprocessing.scale(automobile_features[['horsepower']])

X_train, x_test, Y_train, y_test = train_test_split(automobile_features, automobile_target, test_size=0.2, random_state=0)

dtype = torch.float

X_train_tensor = torch.tensor(X_train.values, dtype=dtype)
x_test_tensor = torch.tensor(x_test.values, dtype=dtype)
Y_train_tensor = torch.tensor(Y_train.values, dtype=dtype)
y_test_tensor = torch.tensor(y_test.values, dtype=dtype)

X_train_tensor.shape
x_test_tensor.shape
Y_train_tensor.shape
y_test_tensor.shape

input_size = 26
output_size = 1
hidden_size = 100
loss_fn = torch.nn.MSELoss()

learning_rate = 0.0001

model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size), 
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size),
        )

for iter in range(1000):
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, Y_train_tensor)
    
    if iter % 100 == 0:
        print(iter, loss.item())
        
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    

