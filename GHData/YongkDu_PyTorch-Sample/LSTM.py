# In[1]:

# Predict the price of the stock of a company

"""## [Step 1] Import basic libraries"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import torch.nn as nn
import torch


"""## [Step 2] Loading the dataset"""

df_train = pd.read_csv('train.csv')['price'].values
df_train = df_train.reshape(-1, 1)
df_test = pd.read_csv('test.csv')['price'].values
df_test = df_test.reshape(-1, 1)

dataset_train = np.array(df_train)
dataset_test = np.array(df_test)


# In[2]:


"""# [Step 3] Pre process your data (no restrictions) """

## Pre process your data in any way you want
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.fit_transform(dataset_test)
# from sklearn import preprocessing
# dataset_train = preprocessing.normalize([dataset_train]).reshape(-1, 1)
# dataset_test = preprocessing.normalize([dataset_test]).reshape(-1, 1)
##########################################


# In[3]:


"""### We create the X_train and Y_train from the dataset train
We take a price on a date as y_train and save the previous 50 closing prices as x_train
"""

trace_back = 50
def create_dataset(df):
    x, y = [], []
    for i in range(trace_back, len(df)):
        # x.append(df[i-trace_back:i, 0])
        x.append(df[i-trace_back:i, 0])
        y.append([df[i, 0]])
    return np.array(x),np.array(y)

x_train, y_train = create_dataset(dataset_train)

x_test, y_test = create_dataset(dataset_test)

x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)


# In[4]:


"""## [Step 4] Build your RNN model
### You are expect to change the content in the below cell and add your own cells

1. You have to design a RNN model that takes in your x_train and do prediction on x_test
2. Your model should be able to predict on x_test using model.predict(x_test)
3. Do not use any pretrained model.
"""
import torch.nn.functional as F
## Your RNN model goes here

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(50, 32, 1, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
#         self.fc2 = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.2)
        

    def predict(self, x):
        h0 = torch.zeros(2, 32).requires_grad_()
        c0 = torch.zeros(2, 32).requires_grad_()
        out, (h1, c1) =  self.lstm(x,(h0,c0))
        out = self.fc1(out)
        out = self.drop(out)
#         out = F.relu(out)
#         out = self.fc2(out)
#         out = F.relu(out)
#         out = self.drop(out)
        out = self.fc3(out)
        return out


# In[5]:


num_epochs = 630
batch_size = 100
num_batch = len(x_train)//batch_size

model = LSTM()
criterion = torch.nn.L1Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


# In[6]:


import time

start_time = time.time()
lstm = []

for e in range(num_epochs):
    p = np.random.permutation(range(len(x_train)))
    X,Y = x_train[p],y_train[p]
    X = torch.from_numpy(x_train).type(torch.Tensor)
    Y = torch.from_numpy(y_train).type(torch.Tensor)
    for i in range(num_batch):
        x = X[i*batch_size:i*batch_size+batch_size]
        y = Y[i*batch_size:i*batch_size+batch_size]
        y_pred = model.predict(x)
        loss = criterion(y_pred, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    if e % 2 == 0:
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions.detach().numpy())
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse_score = mean_squared_error([x[0] for x in y_test_scaled], [x[0] for x in predictions], squared=False)

# training_time = time.time()-start_time
# print("Training time: {}".format(training_time))


# In[7]:


##########################################

"""## [Step 5]: Predictions on X_test
DO NOT change the below code
"""

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions.detach().numpy())

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

"""## [Step 6]: Checking the Root Mean Square Error on X_test"""

rmse_score = mean_squared_error([x[0] for x in y_test_scaled], [x[0] for x in predictions], squared=False)
print("RMSE:",rmse_score)
