import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


lr = 0.01
epochs = 100


class DiabetesData(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


filepath = r'../diabetes.csv'
train_set = DiabetesData(filepath)
train_loader = DataLoader(dataset=train_set,
                          shuffle=True,
                          batch_size=32,
                          num_workers=0)


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.linear2 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x


model = LogisticRegression()


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(epochs):
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        x, y = data[0], data[1]
        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    print('epoch={}, loss={}'.format(epoch+1, epoch_loss/(i+1)))
