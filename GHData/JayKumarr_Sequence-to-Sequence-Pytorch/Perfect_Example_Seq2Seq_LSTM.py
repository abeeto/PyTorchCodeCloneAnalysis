"""
This program takes series of input and predict one output. This is designed for predicting one-dimensional value
"""
# This prorgram has 4 Sections
# Section 1: Reading the data from CSV into DataFrame
# Section 2: Split the data into sequences
# Section 3: Normalizing the data
# Section 4: Combine the all the features into one dataframe (n_samples, timestamps, instance_dim_size)
# ------------ Split the data into TRAINING and TESTING
# Section 5: LSTM Class
# Section 6: Training the model

import pandas as pd
import numpy as np
import torch.nn as nn
import torch

# -----------SECTION 1 -----------------------------
df_ = pd.read_csv("data/PRICE_AND_DEMAND_202201_NSW1.csv")
df_["Date Time"] = pd.to_datetime(df_["SETTLEMENTDATE"])
df_['Day'] = df_['Date Time'].dt.day
df_['Month'] = df_['Date Time'].dt.month
df_['Year'] = df_['Date Time'].dt.year
df_['Hour'] = df_['Date Time'].dt.hour
df_['Minute'] = df_['Date Time'].dt.minute
df_['Demand'] = pd.to_numeric(df_['TOTALDEMAND'], errors ='coerce')
df_['Price'] = pd.to_numeric(df_['RRP'], errors ='coerce')
df_ = df_[['Day', 'Month',  'Year', 'Hour', 'Minute', 'Demand', 'Price']]

# ----------SECTION 2 ---------------------------------
# split data into sequences
seq_length = 48
x0, x1, x2, x3, x4, x5, x6 = [],[],[],[],[],[],[]
y = []
total_points = df_.shape[0]
for i in range(0, total_points-48):
    x0.append(df_.iloc[i:i + seq_length, 0])
    x1.append(df_.iloc[i:i + seq_length, 1])
    x2.append(df_.iloc[i:i + seq_length, 2])
    x3.append(df_.iloc[i:i + seq_length, 3])
    x4.append(df_.iloc[i:i + seq_length, 4])
    x5.append(df_.iloc[i:i + seq_length, 5])
    x6.append(df_.iloc[i:i + seq_length, 6])
    y.append(df_.iloc[i + seq_length, 6])
    a = 10

x0, x1, x2, x3, x4, x5, x6, y = np.array(x0), np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6), np.array(y)
print("y.shape ", y.shape)
y = np.reshape(y, (len(y), 1))
print("y.shape ", y.shape)
print("x0.shape ", x0.shape)

#-----------------SECTION 3----------------------
# normalizing the data between 0 - 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
x0 = scaler.fit_transform(x0)
x1 = scaler.fit_transform(x1)
x2 = scaler.fit_transform(x2)
x3 = scaler.fit_transform(x3)
x4 = scaler.fit_transform(x4)
x5 = scaler.fit_transform(x5)
x6 = scaler.fit_transform(x6)
y = scaler.fit_transform(y)

# -------------SECTION 4-----------------------------------
# combine all features
x_data = np.stack( [x0, x1, x2, x3, x4, x5, x6] , axis=2)
print("x_data.shape : ",x_data.shape)

# Splitting data into train and test
X_train, X_test = x_data[:-480], x_data[-480:]
y_train, y_test = y[:-480], y[-480:]
# -----------------------------------------------------

# ---------- SECTION 5-----------------------
class MyLSTM(nn.Module):
    def __init__(self, input_feature_size, time_steps = 48, output_feature_size = 1, hidden_size = 120, n_layer = 2, dropout = 0.15, device = None):
        super(MyLSTM, self).__init__()
        self.input_feature_size = input_feature_size
        self.timesteps = time_steps
        self.hidden_size = hidden_size
        self.n_layers = n_layer
        self.output_feature_size = output_feature_size

        # LSTM takes input_size and return hidden_size (output)
        self.rnnn = nn.LSTM(input_size=input_feature_size, hidden_size=hidden_size, num_layers=n_layer, dropout=dropout, dtype=torch.float64)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1, dtype=torch.float64)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):

        h_0 = torch.zeros(self.n_layers, self.timesteps, self.hidden_size, dtype=torch.float64).to(device)  # hidden state
        c_0 = torch.zeros(self.n_layers, self.timesteps, self.hidden_size, dtype=torch.float64).to(device)  # internal state
        output, (hn, cn) = self.rnnn(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        out = self.fc(output[:,-1,:])


        return out, hn


#---------------- Section 6 ------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm1 = MyLSTM(input_feature_size=X_train.shape[2], hidden_size=150, output_feature_size=1, device=device).to(device) #our lstm class
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=0.01)

epoches = 10
y_train = torch.from_numpy(y_train).to(device)
X_train = torch.from_numpy(X_train).to(device)
for e in range(epoches):
    output_train, hidden_state = lstm1(X_train)
    loss_train = criterion(output_train, y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    print("epoch {0}/{1} : Loss: {2}".format(e, epoches, loss_train.item()))
