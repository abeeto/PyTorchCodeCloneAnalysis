import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

INPUT_SIZE = 28 * 28
NUM_OF_EPOCHS = 10
NUM_OF_CLASSES = 10


class ModelA(nn.Module):
    BATCH_SIZE = 64

    def __init__(self, input_size):
        super(ModelA, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(nn.Linear(input_size, 100),
                                         nn.ReLU())

        self.hidden_layer_1 = nn.Sequential(nn.Linear(100, 50),
                                            nn.ReLU())

        self.hidden_layer_2 = nn.Linear(50, NUM_OF_CLASSES)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        return F.log_softmax(self.hidden_layer_2(x), dim=1)


class ModelB(nn.Module):
    BATCH_SIZE = 64
    DROPOUT_PROBABILITY = 0.3

    def __init__(self, input_size):
        super(ModelB, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(nn.Linear(input_size, 100),
                                         nn.ReLU())

        self.hidden_layer_1 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                            nn.Linear(100, 50),
                                            nn.ReLU())

        self.hidden_layer_2 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                            nn.Linear(50, NUM_OF_CLASSES))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        return F.log_softmax(self.hidden_layer_2(x), dim=1)


class ModelC(nn.Module):
    BATCH_SIZE = 64

    def __init__(self, input_size):
        super(ModelC, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(nn.Linear(input_size, 100),
                                         nn.BatchNorm1d(100),
                                         nn.ReLU())

        self.hidden_layer_1 = nn.Sequential(nn.Linear(100, 50),
                                            nn.BatchNorm1d(50),
                                            nn.ReLU())

        self.hidden_layer_2 = nn.Linear(50, NUM_OF_CLASSES)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        return F.log_softmax(self.hidden_layer_2(x), dim=1)


class ModelD(nn.Module):
    BATCH_SIZE = 64
    DROPOUT_PROBABILITY = 0.3

    def __init__(self, input_size):
        super(ModelD, self).__init__()
        self.input_size = input_size
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer_1 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                        nn.Linear(7 * 7 * 32, 100),
                                        nn.BatchNorm1d(100),
                                        nn.ReLU())

        self.fc_layer_2 = nn.Sequential(nn.Dropout(self.DROPOUT_PROBABILITY),
                                        nn.Linear(100, NUM_OF_CLASSES))

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer_1(x)
        return F.log_softmax(self.fc_layer_2(x), dim=1)


def save_test_prediction(output_path, test_predictions):
    with open(output_path, "wb") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))


def split_train_set(train_set, batch_size):
    num_train = len(train_set)
    validation_percent = 80
    validation_slice_size = int((validation_percent / 100.0) * num_train)

    validation_idx = np.random.choice(num_train, size=validation_slice_size, replace=False)
    train_idx = list(set(np.arange(num_train)) - set(validation_idx))

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_idx))

    validation_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    sampler=SubsetRandomSampler(validation_idx))

    return train_loader, validation_loader


def plot_average_loss(train_average_loss_per_epoch, validation_average_loss_per_epoch):
    plt.plot(range(1, NUM_OF_EPOCHS + 1), train_average_loss_per_epoch, label="train")
    plt.plot(range(1, NUM_OF_EPOCHS + 1), validation_average_loss_per_epoch, label="validation")
    plt.xlabel("epoch number")
    plt.ylabel("average loss")
    plt.legend()
    plt.show()


def train(net, optimizer, train_loader, validation_loader):
    train_average_loss_per_epoch = []
    validation_average_loss_per_epoch = []

    for epoch in range(NUM_OF_EPOCHS):

        train_correct_count = 0
        train_loss = 0.0

        net.train()
        for data in train_loader:
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = F.nll_loss(outputs, labels, size_average=False)
            train_correct_count += outputs.max(dim=1)[1].eq(labels).sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        validation_correct_count = 0
        validation_loss = 0.0
        net.eval()
        for data in validation_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = F.nll_loss(outputs, labels, size_average=False)
            validation_correct_count += outputs.max(dim=1)[1].eq(labels).sum()

            validation_loss += loss.item()

        train_average_loss = train_loss / len(train_loader.sampler)
        validation_average_loss = validation_loss / len(validation_loader.sampler)

        train_average_loss_per_epoch.append(train_average_loss)
        validation_average_loss_per_epoch.append(validation_average_loss)
        print "[%d train] accuracy: %.3f, average loss: %.3f" % \
              (epoch + 1, float(train_correct_count) / len(train_loader.sampler), train_average_loss)
        print "[%d validation] accuracy: %.3f, average loss: %.3f" % \
              (epoch + 1, float(validation_correct_count) / len(validation_loader.sampler), validation_average_loss)

    plot_average_loss(train_average_loss_per_epoch, validation_average_loss_per_epoch)
    print "Finished Training"


def test(net, test_loader):
    predictions = []
    test_correct_count = 0
    test_loss = 0.0
    net.eval()
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        loss = F.nll_loss(outputs, labels, size_average=False)
        predictions.extend(outputs.max(dim=1)[1].data.tolist())
        test_correct_count += outputs.max(dim=1)[1].eq(labels).sum()

        test_loss += loss.item()

    print "[test] accuracy: %.3f, average loss: %.3f" % (float(test_correct_count) / len(test_loader.dataset),
                                                         test_loss / len(test_loader.dataset))

    save_test_prediction("test.pred", predictions)


if __name__ == "__main__":
    root = './resources'

    if not os.path.exists(root):
        os.mkdir(root)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)

    train_loader, validation_loader = split_train_set(train_set, batch_size=ModelD.BATCH_SIZE)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=64,
                                              shuffle=False)
    net = ModelD(INPUT_SIZE)

    optimizer = optim.Adam(net.parameters())

    train(net, optimizer, train_loader, validation_loader)
    test(net, test_loader)
