import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def preprocessing():
    df = pd.read_csv('./data/titanic/titanic.csv')
    df = df.dropna()
    df['Sex'] = df['Sex'].map({'female' : 1, 'male' : 0}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S' : 2, 'C' : 1, 'Q' : 0}).astype(int)
    df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], 1)
    df.to_csv('./data/titanic/titanic_processed.csv', index=False, header=False)

class Dataset(Dataset):
    def __init__(self):
        preprocessing()
        xy = np.loadtxt('./data/titanic/titanic_processed.csv', delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])
        self.len = len(xy)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def __len__(self):
        return self.len

dataset = Dataset()
train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(7, 4)
        self.l2 = torch.nn.Linear(4, 1)
        self.sigmoid = F.sigmoid

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        return out2

model = Model()
criterion = torch.nn.BCELoss(size_average=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    y_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        print("epoch: %d, iter: %d, loss: %.3f" %(epoch, i, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_loss = loss.data[0]
    plt.plot(epoch, y_loss, 'r.')

plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
