####################################################################
# Neural Network with multiple dimensional input
####################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Plot the data
def plot_decision_regions_2class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()


# Calculate the accuracy
def accuracy(model, data_set):
    return np.mean(data_set.y.view(-1).numpy()
                   == (model(data_set.x)[:, 0] > 0.5).numpy())


# Define the class Net with one hidden layer
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.sigmoid(self.linear1(x))
        yhat = torch.sigmoid(self.linear2(y1))
        return yhat


# Define dataset class
class XOR_Data(Dataset):
    def __init__(self, N_s=100):
        self.x = torch.zeros(N_s, 2)
        self.y = torch.zeros(N_s, 1)
        for i in range(N_s // 4):  # //: floor operator
            self.x[i, :] = torch.Tensor([0.0, 0.0])
            self.y[i, 0] = torch.Tensor([0.0])

            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([1.0])

            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([1.0])

            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([0.0])

            self.x = self.x + 0.01 * torch.randn(N_s, 2)
        self.len = N_s

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the data
    def plot_stuff(self):
        plt.figure()
        plt.plot(self.x[self.y[:, 0] == 0, 0].numpy(), self.x[self.y[:, 0] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:, 0] == 1, 0].numpy(), self.x[self.y[:, 0] == 1, 1].numpy(), 'ro', label="y=1")
        plt.legend()


# Create dataset
data_set = XOR_Data()
data_set.plot_stuff()
model = Net(2, 2, 1)
learning_rate = 0.1
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=1)


# Create function to train model
def train(data_set, model, criterion, train_loader, optimizer, iter=5):
    COST = []
    ACC = []
    for i in range(iter):
        total = 0
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        ACC.append(accuracy(model, data_set))
        COST.append(total)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
    return COST


LOSS12 = train(data_set, model, criterion, train_loader, optimizer, iter=500)
plot_decision_regions_2class(model, data_set)




