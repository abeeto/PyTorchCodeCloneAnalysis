import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy[:, 0:-1])
        self.y_data = torch.tensor(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 2)
        self.l4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        ret = self.sigmoid(self.l4(out3))
        return ret


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
model = MyModel()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def learn(x, y):
    y_pred = model(x)
    learn_loss = criterion(y_pred, y)
    optimizer.zero_grad()
    learn_loss.backward()
    optimizer.step()
    return learn_loss.data.item()


def run(total=10000):
    for epoch in range(total):
        for idx, data in enumerate(train_loader):
            inputs, labels = map(Variable, data)
            print(inputs, type(inputs))
            loss = learn(inputs, labels)
            print(f"Loss in epoch {epoch + 1} idx {idx} : {loss}", end="\r")
    print()


def check():
    return ((model(dataset.x_data) > 0.5) == (dataset.y_data == 1.0)).sum().item() / len(dataset)
