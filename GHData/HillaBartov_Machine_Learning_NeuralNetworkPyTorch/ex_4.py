import sys
import numpy as np
from scipy import stats
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.utils import data

IMAGE_SIZE = 784
BATCH_SIZE = 64


class DataSet(data.Dataset):
    def __init__(self, x, y):
        self.x_data = t.tensor(x, dtype=t.float32)
        self.y_data = t.tensor(y, dtype=t.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class NeuralNetAB(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetAB, self).__init__()
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # reshape
        x = x.view(-1, IMAGE_SIZE)
        # forward through the layers
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return softmax
        return F.log_softmax(self.fc2(x), dim=1)


class NeuralNetC(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetC, self).__init__()
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # reshape
        x = x.view(-1, IMAGE_SIZE)
        # forward through the layers
        drop = nn.Dropout(0.4)
        x = F.relu(self.fc0(x))
        x = drop(x)
        x = F.relu(self.fc1(x))
        x = drop(x)
        x = F.relu(self.fc2(x))
        x = drop(x)
        # return softmax
        return F.log_softmax(x, dim=1)


class NeuralNetD(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetD, self).__init__()
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # reshape
        x = x.view(-1, IMAGE_SIZE)
        batch_norm = nn.BatchNorm1d(IMAGE_SIZE, affine=False)
        x = batch_norm(x)
        # forward through the layers
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # return softmax
        return F.log_softmax(x, dim=1)


class NeuralNetE(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetE, self).__init__()
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        # reshape
        x = x.view(-1, IMAGE_SIZE)
        # forward through the layers
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # return softmax
        return F.log_softmax(x, dim=1)


class NeuralNetF(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetF, self).__init__()
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        # reshape
        x = x.view(-1, IMAGE_SIZE)
        # forward through the layers
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        # return softmax
        return F.log_softmax(x, dim=1)


def train(optimizer, model):
    model.train()
    avg_loss = 0
    correct = 0
    for batch_idx, (x, label) in enumerate(train_loader):
        optimizer.zero_grad()
        # matrix dims- batch,10
        y_hat = model(x)
        # calculate loss
        loss = F.nll_loss(y_hat, label)
        avg_loss += loss.data.item()
        # back prop - compute gradients
        loss.backward()
        # update model
        optimizer.step()
        pred = y_hat.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).cpu().sum()


def validate(model):
    # test mode
    model.eval()
    validation_loss = 0
    correct = 0
    with t.no_grad():
        for x, y in validation_loader:
            y_hat = model(x)
            validation_loss += F.nll_loss(y_hat, y).item()
            # get indexes of max probabilities in batch
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).cpu().sum()

    avg_loss = validation_loss / validation_size
    acc = 100 * correct / validation_size
    # print(f'epoch {epoch + 1} Average loss: {avg_loss:.4f} accuracy: {acc:.0f}%')


def test(model):
    # test mode
    model.eval()
    with t.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            pred = t.max(y_hat, 1).indices.numpy()
            for label in pred:
                y_hats.append(int(label))


def normalize(values):
    values /= 255
    return stats.zscore(values, axis=None)


# Loading Files
train_x = np.loadtxt(sys.argv[1])
train_y = np.loadtxt(sys.argv[2])
test_x = np.loadtxt(sys.argv[3])

# Normalize train and test values
train_x = normalize(train_x)
test_x = normalize(test_x)

data_set = DataSet(train_x, train_y)
test_set = DataSet(test_x, np.zeros(5000))
# Split data set into 80% Train and 20% Validation
train_size = round(data_set.__len__() * 0.8)
validation_size = data_set.__len__() - train_size
train_set, validation_set = data.random_split(data_set, [train_size, validation_size])

# Train validation and test set loaders
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model D
model = NeuralNetD(IMAGE_SIZE)
# Optimizer
optimizer_SGD = t.optim.SGD(model.parameters(), lr=0.1)
# model's predictions
y_hats = []

for epoch in range(10):
    train(optimizer_SGD, model)
    validate(model)
test(model)
np.savetxt('test_y', np.array(y_hats), fmt="%i")
