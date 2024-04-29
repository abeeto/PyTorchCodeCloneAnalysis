import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class TwoLayersModel(nn.Module):
    def __init__(self, image_size):
        super(TwoLayersModel, self).__init__()
        self.image_size = image_size
        self.first_layer = nn.Linear(image_size, 100)
        self.second_layer = nn.Linear(100, 50)
        self.third_layer = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.third_layer(x)
        return F.log_softmax(x, dim=1)


class DropoutModel(nn.Module):
    def __init__(self, image_size):
        super(DropoutModel, self).__init__()
        self.image_size = image_size
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.first_layer = nn.Linear(image_size, 100)
        self.second_layer = nn.Linear(100, 50)
        self.third_layer = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.first_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.second_layer(x))
        x = self.dropout2(x)
        x = self.third_layer(x)
        return F.log_softmax(x, dim=1)


class BatchNormalization(nn.Module):
    def __init__(self, image_size):
        super(BatchNormalization, self).__init__()
        self.image_size = image_size
        self.first_layer = nn.Linear(image_size, 100)
        self.second_layer = nn.Linear(100, 50)
        self.third_layer = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    # Batch normalization - before activation fucntion.
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.first_layer(x)
        x = F.relu(self.bn1(x))
        x = self.second_layer(x)
        x = F.relu(self.bn2(x))
        x = self.third_layer(x)
        return F.log_softmax(x, dim=1)


class FiveLayersModelWithReLu(nn.Module):
    def __init__(self, image_size):
        super(FiveLayersModelWithReLu, self).__init__()
        self.image_size = image_size
        self.first_layer = nn.Linear(image_size, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, 10)
        self.fourth_layer = nn.Linear(10, 10)
        self.fifth_layer = nn.Linear(10, 10)
        self.sixth_layer = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = F.relu(self.third_layer(x))
        x = F.relu(self.fourth_layer(x))
        x = F.relu(self.fifth_layer(x))
        x = self.sixth_layer(x)
        return F.log_softmax(x, dim=1)


class FiveLayersModelWithReSigmoid(nn.Module):
    def __init__(self, image_size):
        super(FiveLayersModelWithReSigmoid, self).__init__()
        self.image_size = image_size
        self.first_layer = nn.Linear(image_size, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, 10)
        self.fourth_layer = nn.Linear(10, 10)
        self.fifth_layer = nn.Linear(10, 10)
        self.sixth_layer = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.first_layer(x))
        x = torch.sigmoid(self.second_layer(x))
        x = torch.sigmoid(self.third_layer(x))
        x = torch.sigmoid(self.fourth_layer(x))
        x = torch.sigmoid(self.fifth_layer(x))
        x = self.sixth_layer(x)
        return F.log_softmax(x, dim=1)

class BestModel(nn.Module):
    def __init__(self, image_size):
        super(BestModel, self).__init__()
        self.image_size = image_size
        self.first_layer = nn.Linear(image_size, 700)
        self.second_layer = nn.Linear(700, 50)
        self.third_layer = nn.Linear(50, 10)
        self.fourth_layer = nn.Linear(10, 10)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(700)
        self.bn2 = nn.BatchNorm1d(50)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.first_layer(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.second_layer(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)
        x = self.third_layer(x)
        x = F.relu(self.bn3(x))
        x = self.fourth_layer(x)
        return F.log_softmax(x, dim=1)

# Train the model.
def train(train_loader, epoch, model, optimizer):
    model.train()
    for i in range(epoch):
        for (data, label) in train_loader:
            optimizer.zero_grad()
            # Predict target
            output = model(data)
            # Calculate loss
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()


def predict(model, test_x):
    model.eval()
    y_hats = []
    with torch.no_grad():
        # For all x in text_x, predict the target.
        for x in test_x:
            output = model(x)
            p = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            y_hats.append(p)
    return y_hats


def test(test_loader, model):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            loss += F.nll_loss(output, y, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).cpu().sum().item()
        loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# This function show the results of train & validation in one graph.
def show_graph(train, validation):
    x = list(range(15))
    plt.plot(x, train, c='blue', label='Train')
    plt.plot(x, validation, c='green', label='Validation')
    plt.ylabel('Loss average')
    plt.xlabel('Epoch')
    plt.xticks(x, x)
    plt.title('Best Model -- Loss average')
    plt.legend()
    plt.show()


def graph_lists(train_loader, test_loader, epoch, model, optimizer):
    loss_list_train = []
    accuracy_list_train = []
    loss_list_valid = []
    accuracy_list_valid = []
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    for i in range(epoch):
        model.train()
        correct_sum_train = 0
        loss_sum_train = 0
        for (data, label) in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label, reduction='sum')
            loss_sum_train += loss.item()
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct_sum_train += pred.eq(label.view_as(pred)).cpu().sum().item()
        loss_list_train.append(loss_sum_train / train_size)
        accuracy_list_train.append(correct_sum_train / train_size)

        correct_sum_valid = 0
        loss_sum_valid = 0
        for x, y in test_loader:
            output = model(x)
            loss_sum_valid += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct_sum_valid += pred.eq(y.view_as(pred)).cpu().sum().item()
        loss_list_valid.append(loss_sum_valid / test_size)
        accuracy_list_valid.append(correct_sum_valid / test_size)

    return loss_list_train, accuracy_list_train, loss_list_valid, accuracy_list_valid


train_x, train_y, test_x, outfile_name = sys.argv[1:5]
# Normalization the data.
X = np.loadtxt(train_x) / 255
Y = np.loadtxt(train_y)
test_x = np.loadtxt(test_x) / 255

# Convert to torch arrays.
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).long()
test_x = torch.from_numpy(test_x).float()

XY = TensorDataset(X, Y)
train_loader = torch.utils.data.DataLoader(XY, batch_size=64, shuffle=True)

# Train -- Validation partition.
# train_data, test_data = torch.utils.data.random_split(XY, [int(0.8 * len(XY)), int(0.2 * len(XY))])

# Run best model and print the results to "test_y" file.
epoch = 15
best_model = BestModel(image_size=28 * 28)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
train(train_loader, epoch, best_model, optimizer)
predictions = predict(best_model, test_x)

with open(outfile_name, "w") as f:
    for i in range(len(test_x) - 1):
        f.write(str(predictions[i].item()) + '\n')
    f.write(str(predictions[len(test_x) - 1].item()))
