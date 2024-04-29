#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:14:30 2020

@author: siddharthsmac
"""

import numpy as np
import pandas as pd

df = pd.read_csv('/users/siddharthsmac/downloads/diabetes.csv')

import seaborn as sns

df['Outcome'] = np.where(df['Outcome'] == 1, 'Diabetic', 'Non-Diabetic')
sns.pairplot(df, hue = 'Outcome')

df = pd.read_csv('/users/siddharthsmac/downloads/diabetes.csv')

X = df.drop('Outcome', axis = 1).values
y = df['Outcome'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import torch
import torch.nn as nn
import torch.nn.functional as F

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class ANN_Model(nn.Module):
    def __init__(self, input_features = 8, hidden1 = 20, hidden2 = 20, out_features = 2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)
    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

torch.manual_seed(20)

model = ANN_Model()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs  = 1000

final_losses = []
for i in range(epochs):
    i = i+1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)
    if i%10 ==1:
        print('Epoch number: {} and the loss: {}'.format(i , loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(epochs), final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss variation')
plt.show()

predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
        
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize = (10,6))
sns.heatmap(cm, annot = True)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Confusion Matrix')
plt.show()

score = accuracy_score(y_test, predictions)
print(score)






torch.save(model, 'diabetes_prediction.pt')

model = torch.load('diabetes_prediction.pt')
model.eval()

## Predict new data

lst1 = [6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
new_data = torch.tensor(lst1)

with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax().item())

