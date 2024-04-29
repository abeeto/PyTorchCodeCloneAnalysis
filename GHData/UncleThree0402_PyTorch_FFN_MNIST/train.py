import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy
import torchvision
import torchvision.transforms as transforms
import time

import random

import utils

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Import dataset
train_dataset = torchvision.datasets.MNIST(root="./MNIST", train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.MNIST(root="./MNIST", train=False, download=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))

# Process
dataT = torch.concat((train_dataset.data, test_dataset.data), 0)
labelsT = torch.concat((train_dataset.targets, test_dataset.targets), 0)
dataT = dataT.float()
labelsT = labelsT.long()
dataT = nn.Flatten()(dataT)

utils.plot_img_example(dataT, labelsT, (28, 28), "MNIST_numbers_Images")
utils.plot_number_example(dataT, labelsT, "Data_before_normalize")
dataT = dataT / torch.max(dataT)
utils.plot_number_example(dataT, labelsT, "Data_after_normalize")

train_dl, valid_dl, test_dl = utils.data_2_data_dl(dataT, labelsT)

# Count of labels
count = torch.unique(labelsT, return_counts=True)
plt.bar(count[0], count[1])
plt.xticks(count[0])
plt.savefig(f'./Photo/count_labels.png')
plt.show()


# Net

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(784, 64)

        self.bn1 = nn.BatchNorm1d(64)

        self.do1 = nn.Dropout(0.2)

        self.ll1 = nn.Linear(64, 32)

        self.bn2 = nn.BatchNorm1d(32)

        self.do2 = nn.Dropout(0.2)

        self.ll2 = nn.Linear(32, 32)

        self.bn3 = nn.BatchNorm1d(32)

        self.do3 = nn.Dropout(0.2)

        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU()(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.ll1(x)
        x = nn.LeakyReLU()(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.ll2(x)
        x = nn.LeakyReLU()(x)
        x = self.bn3(x)
        x = self.do3(x)
        x = self.output(x)
        return x


# Create Tool

def create_mnist_net(lr, wd, lr_step, lr_gamma):
    net = MNISTNet()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    return net, loss_fn, optimizer, lr_scheduler


def train(train_dl, valid_dl, model, loss_fn, optimizer, lr_scheduler, epochs):
    time_train = time.process_time()
    best_model = {"Accuracy": 0, "net": None}

    train_losses = []
    train_accuracies = []

    valid_losses = []
    valid_accuracies = []

    lr_rate = []

    for epoch in range(epochs):
        time_epoch = time.process_time()

        train_batch_losses = []
        train_batch_accuracies = []

        valid_batch_losses = []
        valid_batch_accuracies = []

        for X, y in train_dl:
            yHat_train = model(X)

            train_loss = loss_fn(yHat_train, y)
            train_batch_losses.append(train_loss)

            train_accuracy = utils.accuracy(yHat_train, y)
            train_batch_accuracies.append(train_accuracy)

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        lr_scheduler.step()
        lr_rate.append(lr_scheduler.get_last_lr()[0])

        train_epoch_loss = torch.mean(torch.tensor(train_batch_losses))
        train_epoch_accuracy = torch.mean(torch.tensor(train_batch_accuracies))

        train_losses.append(train_epoch_loss.detach())
        train_accuracies.append(train_epoch_accuracy)

        model.eval()

        with torch.no_grad():
            for X, y in valid_dl:
                yHat_valid = model(X)

                valid_loss = loss_fn(yHat_valid, y)
                valid_batch_losses.append(valid_loss)

                valid_accuracy = utils.accuracy(yHat_valid, y)
                valid_batch_accuracies.append(valid_accuracy)

        model.train()

        valid_epoch_loss = torch.mean(torch.tensor(valid_batch_losses))
        valid_epoch_accuracy = torch.mean(torch.tensor(valid_batch_accuracies))

        valid_losses.append(valid_epoch_loss.detach())
        valid_accuracies.append(valid_epoch_accuracy)

        if valid_epoch_accuracy > best_model["Accuracy"]:
            best_model["Accuracy"] = valid_epoch_accuracy;
            best_model["net"] = copy.deepcopy(model.state_dict())

        print(
            f"Epoch : {epoch + 1}, Train_loss : {train_epoch_loss:.4f}, Train_accuracy : {train_epoch_accuracy:.4f}, Valid_loss : {valid_epoch_loss:.4f}, Valid_accuracy : {valid_epoch_accuracy:.4f}, Time : {(time.process_time() - time_epoch):.2f} sec")

    # Save Best Model
    torch.save(best_model["net"], "train.pt")

    print(f"Total Time : {(time.process_time() - time_train):.2f} sec")

    return train_losses, train_accuracies, valid_losses, valid_accuracies, lr_rate


net, loss_fn, optimizer, lr_scheduler = create_mnist_net(0.001, 0.02, 5, 0.5)
train_losses, train_accuracies, valid_losses, valid_accuracies, lr_rate = train(train_dl, valid_dl, net, loss_fn,
                                                                                optimizer, lr_scheduler, 50)

# Load best model
net.load_state_dict(torch.load("train.pt"))

# plot

utils.plot_loss(train_losses, valid_losses)
utils.plot_accuracy(train_accuracies, valid_accuracies)
utils.plot_lr(lr_rate)
utils.show_performance_binary(train_dl, valid_dl, test_dl, net)
