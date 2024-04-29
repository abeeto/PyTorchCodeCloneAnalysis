import numpy as np
from torch import optim
import torch
import torch.nn as nn
from torch.autograd import Variable

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

Xtr=Variable(torch.FloatTensor(Xtr))
Ytr=Variable(torch.FloatTensor(Ytr.reshape(-1, 1)))

model = nn.Sequential(nn.Linear(8, 30), nn.ReLU(), nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1),
                      #nn.LogSoftmax(dim=1))
                      nn.ReLU())

criterion = RMSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loss =[]
val_loss=[]
epochs = 70000

for e in range(epochs):
    optimizer.zero_grad()
    output_t = model(Xtr)
    loss = criterion(output_t, Ytr)
    print('epoch: ', e, ' loss: ', loss.item())
    loss.backward()
    optimizer.step()

    model.train()
    train_loss.append(loss.item())


torch.save(model, 'myModel.pth')