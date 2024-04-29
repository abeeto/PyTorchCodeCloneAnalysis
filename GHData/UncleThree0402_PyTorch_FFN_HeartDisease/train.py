import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy

import random

import utils

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Data Import

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, sep=",", header=None)
data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                "thal", "num"]
data = data.replace('?', np.nan).dropna()

data_keys = data.keys()

for d in data_keys:
  num = pd.to_numeric(data[d])
  data[d] = num

print(f"Data before normalized:\n{data.describe()}")

utils.box_plot(data, "Before Normalize")

# Data Normalize

keys = data.keys()
keys = keys.drop(["sex", "fbs", "exang", "num"])

for d in keys:
    num = pd.to_numeric(data[d])
    data[d] = num

data[keys] = data[keys].apply(stats.zscore)

data["num"][data["num"] > 0] = 1

print(f"Data before normalized:\n{data.describe()}")

utils.box_plot(data, "After Normalize")

# Visualize labels

count = pd.value_counts(data["num"])
plt.bar(list(count.keys()), count)
plt.xticks(np.arange(0, 2), ["No", "Yes"])
plt.title("Count of labels")
plt.show()

# To Tensor

data_keys = data.keys()
data_keys = data_keys.drop("num")

dataT = torch.tensor(data[data_keys].values).float()
labelsT = torch.tensor(data["num"].values).float()
labelsT = labelsT[:, None]

train_dl, valid_dl, test_dl = utils.data_2_data_dl(dataT, labelsT)


# Net

class HeartNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(13, 20)

        self.ll1 = nn.Linear(20, 30)

        self.d1 = nn.Dropout(0.2)

        self.ll2 = nn.Linear(30, 30)

        self.d2 = nn.Dropout(0.2)

        self.ll3 = nn.Linear(30, 13)

        self.d3 = nn.Dropout(0.2)

        self.output = nn.Linear(13, 1)

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU()(x)
        x = self.d1(x)
        x = self.ll1(x)
        x = nn.LeakyReLU()(x)
        x = self.d2(x)
        x = self.ll2(x)
        x = nn.LeakyReLU()(x)
        x = self.d3(x)
        x = self.ll3(x)
        x = nn.LeakyReLU()(x)
        x = self.output(x)
        return x


# Loss_fn optimizer lr_scheduler

def create_heart_net(lr, wd, lr_step, lr_gamma):
    net = HeartNet()

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    return net, loss_fn, optimizer, lr_scheduler


# Train

def train(train_dl, valid_dl, model, loss_fn, optimizer, lr_scheduler, epochs):
    best_model = {"Accuracy": 0, "net": None}

    train_losses = []
    train_accuracies = []

    valid_losses = []
    valid_accuracies = []

    lr_rate = []

    for epoch in range(epochs):

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
            f"Epoch : {epoch + 1}, Train_loss : {train_epoch_loss:.4f}, Train_accuracy : {train_epoch_accuracy:.4f}, Valid_loss : {valid_epoch_loss:.4f}, Valid_accuracy : {valid_epoch_accuracy:.4f}")

    # Save Best Model
    torch.save(best_model["net"], "train.pt")

    return train_losses, train_accuracies, valid_losses, valid_accuracies, lr_rate


net, loss_fn, optimizer, lr_scheduler = create_heart_net(0.0001, 0.015, 10, 0.9)
train_losses, train_accuracies, valid_losses, valid_accuracies, lr_rate = train(train_dl, valid_dl, net, loss_fn,
                                                                                optimizer, lr_scheduler, 200)

# Load best model
net.load_state_dict(torch.load("train.pt"))

# plot

utils.plot_loss(train_losses, valid_losses)
utils.plot_accuracy(train_accuracies, valid_accuracies)
utils.plot_lr(lr_rate)
utils.show_performance_binary(train_dl, valid_dl, test_dl, net)
