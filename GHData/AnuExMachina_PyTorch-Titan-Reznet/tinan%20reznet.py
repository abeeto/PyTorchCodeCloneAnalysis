import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.nn.functional as F


one_hot_encoder = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()
scaler = MinMaxScaler()
data = pd.read_csv('titanic.csv')


X = data.iloc[:,1:]
y = data.iloc[:,0]

#X = data.drop('Survived', axis=1)
#y = data['Survived']

X = X.drop('Name', axis=1)

transformed = pd.DataFrame(one_hot_encoder.fit_transform(X['Pclass'].to_numpy().reshape(-1, 1)), columns=one_hot_encoder.get_feature_names(['Pclass']))

#transformed = pd.get_dummies(X['Pclass'])
X = pd.concat([X.drop('Pclass', axis=1), transformed], axis=1)

X['Sex'] = label_encoder.fit_transform(X['Sex'])

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train.values.reshape(-1, 1))
y_test = torch.Tensor(y_test.values.reshape(-1, 1))

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)

X.shape
y.shape

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_latent):
        super().__init__()
        self.dense1 = nn.Linear(n_input, n_latent)
        self.dense2 = nn.Linear(n_latent, n_input)
        self.layer_norm = nn.LayerNorm(8)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x1 = F.gelu(self.dense1(x))
        x1 = self.dropout(x1)
        x1 = F.gelu(self.dense2(x1))
        return self.layer_norm(x1 + x) 
        





class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.res_block1 = ResidualBlock(8, 128)
        self.res_block2 = ResidualBlock(8, 128)
        self.res_block3 = ResidualBlock(8, 128)
        self.res_block4 = ResidualBlock(8, 128)
        self.dense = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = F.sigmoid(self.dense(x))
        
        return x


neural_network = NeuralNetwork()
#neural_network(X)

loss_obj = torch.nn.BCELoss()
optimizer = torch.optim.Adam(neural_network.parameters())


for i in range(70):
    neural_network.train()
    optimizer.zero_grad()
    for X, y in dataset_train:
        y_pred = neural_network(X)
        loss = loss_obj(y_pred, y)
        loss.backward()
        optimizer.step()



neural_network.eval()
y_pred = neural_network(X_test)

accuracy_score(y_test.detach().numpy(), np.round(y_pred.detach().numpy(), 0))