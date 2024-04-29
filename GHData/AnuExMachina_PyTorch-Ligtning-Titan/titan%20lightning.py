import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer


one_hot_encoder = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()
scaler = MinMaxScaler()
data = pd.read_csv('titanic.csv')


X = data.iloc[:,1:]
y = data.iloc[:,0]

#X = data.drop('Survived', axis=1)
#y = data['Survived']

X = X.drop('Name', axis=1)
#spagetki one hot encoder.
transformed = pd.DataFrame(one_hot_encoder.fit_transform(X['Pclass'].to_numpy().reshape(-1, 1)), columns=one_hot_encoder.get_feature_names(['Pclass']))
#to samo z pandasa
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

class NeuralNetwork(LightningModule):
    
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(8, 16)
        self.dense2 = nn.Linear(16, 12)
        self.dense3 = nn.Linear(12, 8)
        self.dense4 = nn.Linear(8, 4)
        self.dense5 = nn.Linear(4, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.dropout1(x)
        x = F.gelu(self.dense2(x))
        x = self.dropout2(x)
        x = F.gelu(self.dense3(x))
        x = self.dropout3(x)
        x = F.gelu(self.dense4(x))
        x = self.dropout4(x)
        x = F.sigmoid(self.dense5(x))
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())  


neural_network = NeuralNetwork()
#neural_network(X)
trainer = Trainer()
trainer.fit(neural_network, dataset_train)

#neural_network.dense1.weight

neural_network.eval()
y_pred = neural_network(X_test)

accuracy_score(y_test.detach().numpy(), np.round(y_pred.detach().numpy(), 0))