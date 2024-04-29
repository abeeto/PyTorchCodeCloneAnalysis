import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

batch_size = 64

def preprocessing():
    df = pd.read_csv('./data/otto/train.csv')
    f = lambda x: int(x[6:]) - 1
    df['target'] = df['target'].apply(f)
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    valid = df[~msk]
    train.to_csv('./data/otto/train_processed.csv', index=False, header=False)
    valid.to_csv('./data/otto/valid_processed.csv', index=False, header=False)

class Dataset(Dataset):
    def __init__(self, valid=False):
        preprocessing()
        if valid: data = './data/otto/valid_processed.csv'
        else: data = './data/otto/train_processed.csv'
        xy = np.loadtxt(data, delimiter=',', skiprows=1, dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, 1:-1])
        y = torch.from_numpy(xy[:, [-1]])
        self.y_data = torch.squeeze(y.type(torch.LongTensor))
        self.len = len(xy)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def __len__(self):
        return self.len

dataset = Dataset()
valid_dataset = Dataset(valid=True)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(93, 50)
        self.l2 = torch.nn.Linear(50, 20)
        self.l3 = torch.nn.Linear(20, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

model = Model()

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        loss = criterion(output, target).data[0]
        test_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'. format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return (correct / len(valid_loader.dataset))

results = []
for epoch in range(1, 31):
    train(epoch)
    accuracy = test()
    results = results + [accuracy]

plt.plot(range(1, 31), results)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("validation accuracy")
plt.show()
