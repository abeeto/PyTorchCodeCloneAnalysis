import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA(nn.Module):
    def __init__(self, input_size):
        super(ModelA, self).__init__()
        self.image_size = input_size
        # 2 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self, input_size):
        super(ModelB, self).__init__()
        self.image_size = input_size
        # 2 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self, input_size):
        super(ModelC, self).__init__()
        self.image_size = input_size
        # 2 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)


class ModelD(nn.Module):
    def __init__(self, input_size):
        super(ModelD, self).__init__()
        self.image_size = input_size
        # 2 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.batch_norm2 = nn.BatchNorm1d(50)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = F.relu(self.batch_norm1(self.fc0(x)))
        x = F.relu(self.batch_norm2(self.fc1(x)))
        x = self.fc2(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    def __init__(self, input_size):
        super(ModelE, self).__init__()
        self.image_size = input_size
        # 5 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.batch_norm1 = nn.BatchNorm2d(100)
        self.batch_norm2 = nn.BatchNorm2d(50)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)


class ModelF(nn.Module):
    def __init__(self, input_size):
        super(ModelF, self).__init__()
        self.image_size = input_size
        # 5 hidden layers
        # matrix size image_size x 1000
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.batch_norm1 = nn.BatchNorm2d(100)
        self.batch_norm2 = nn.BatchNorm2d(50)

    def forward(self, x):
        # reshape - example of 28x28 --> 1x784= image_size
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        # negative log likelihood
        return F.log_softmax(x, dim=1)
