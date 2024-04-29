import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def trim_data(df: pd.DataFrame, train=True):
    if train:
        y = torch.tensor(df['Survived'].astype(np.float32)).view(-1, 1)
        dropped_df = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    else:
        dropped_df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    dropped_df['Embarked'].fillna(dropped_df['Embarked'].mode()[0], inplace=True)
    dropped_df['Age'].fillna(dropped_df['Age'].median(), inplace=True)

    x = torch.tensor(pd.get_dummies(dropped_df,
                                    columns=['Pclass', 'Embarked', 'Sex']).to_numpy(dtype=np.float32))
    if train:
        return x, y
    else:
        return x


class TitanicDataset(Dataset):

    def __init__(self):
        self.x_data, self.y_data = trim_data(pd.read_csv('./data/titanic/train.csv'))

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.y_data.shape[0]


class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(12, 10)
        self.linear2 = torch.nn.Linear(10, 6)
        self.linear3 = torch.nn.Linear(6, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.linear1(x))
        out2 = self.sigmoid(self.linear2(out1))
        return self.sigmoid(self.linear3(out2))


dataset = TitanicDataset()
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = MyModel()
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def learn(x, y):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data.item()


def run(total=1000):
    for epoch in range(total):
        for data in dataloader:
            x_data, y_data = map(Variable, data)
            train_loss = learn(x_data, y_data)
            print(f"Epoch {epoch} : {train_loss}", end="\r")
    print()


def check():
    return ((model(dataset.x_data) > 0.5) == (dataset.y_data == 1.0)).sum().item() / len(dataset)


def answer():
    return np.array(model(trim_data(pd.read_csv('./data/titanic/test.csv'), train=False)) > 0.5, dtype=int)

