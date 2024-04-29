import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, Dataset
from torchvision import transforms
from torchvision import datasets

DATA_SET_SIZE = 60000
DATA_SET_TRAIN = 60000 * 0.8
DATA_SET_VALIDATION = 60000 * 0.2
EPOCHS = 10


def train(model, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_set):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model, set, size_of_data):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in set:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= size_of_data
    current_accuracy = 100. * correct / size_of_data
    # print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #  test_loss, correct, size_of_data,
    #  current_accuracy))
    return test_loss, current_accuracy


def test_x(model, set):
    file = open("test_y", "w")
    model.eval()
    for data in set:
        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        for i in list(pred):
            file.write(str(int(i)) + '\n')
    file.close()


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SecondNet(nn.Module):
    def __init__(self, image_size):
        super(SecondNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(F.relu(self.fc0(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ThirdNet(nn.Module):
    def __init__(self, image_size):
        super(ThirdNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.batchN0 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.batchN1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = F.relu(self.batchN0(x))
        x = self.fc1(x)
        x = F.relu(self.batchN1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ForthNet(nn.Module):
    def __init__(self, image_size):
        super(ForthNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class FifthNet(nn.Module):
    def __init__(self, image_size):
        super(FifthNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                    transform=transforms)
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
    train_set, validation_set = torch.utils.data.random_split(dataset, lengths)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validation_set = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)

    test_x_set = np.loadtxt('test_x')
    test_x_set *= (1 / 255)
    test_x_tensor = torch.FloatTensor(test_x_set)
    test_x_loader = torch.utils.data.DataLoader(test_x_tensor, batch_size=64, shuffle=False)

    test_set = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = ThirdNet(image_size=28 * 28)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, int(EPOCHS) + 1):
        train(model, optimizer)
    test_x(model, test_x_loader)
