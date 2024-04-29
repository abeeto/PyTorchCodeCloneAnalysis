from sklearn.model_selection import train_test_split, GridSearchCV
from torch import optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from skorch import NeuralNetRegressor

Xtr = np.loadtxt("TrainData.csv")
Ytr=np.loadtxt("TrainLabels.csv")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

model = nn.Sequential(nn.Linear(8, 30), nn.ReLU(), nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1),
                      #nn.LogSoftmax(dim=1))
                      nn.Sigmoid())


nnr=NeuralNetRegressor(
    module=model,
    train_split=None,
    criterion=RMSELoss,
    optimizer=torch.optim.SGD
)

Xtr=Variable(torch.FloatTensor(Xtr))
Ytr=Variable(torch.FloatTensor(Ytr.reshape(-1, 1)))


#Cross Validation
distributions = {
    'lr': [0.01, 0.02],
    'max_epochs': [70000, 50000]
}


grid = GridSearchCV(nnr, distributions, cv=5, scoring="neg_mean_squared_error")
grid.fit(Xtr.detach().numpy(), Ytr.detach().numpy())
print("Best Neural Network Params: ", grid.best_params_)






























