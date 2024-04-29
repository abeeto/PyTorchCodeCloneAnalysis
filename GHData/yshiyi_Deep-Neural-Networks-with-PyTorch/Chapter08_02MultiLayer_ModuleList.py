####################################################################
# Multiple Hidden Layer Deep Network
# Building deep neural networks with nn.ModuleList()
####################################################################
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)


# Define the function to plot the diagram
def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label='y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label='y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label='y=2')
    plt.title("decision region")
    plt.legend()


# Create dataset class
class Data(Dataset):
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N*K, D))  # shape is (N*K, D)
        y = np.zeros(N*K)  # shape is (N*K, )
        for j in range(K):
            ix = range(N*j, N*(j+1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + \
                np.random.randn(N) * 0.2  # theta
            # np.c_: add along second axis
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    # Plot the diagram
    def plot_stuff(self):
        plt.figure()
        plt.plot(self.x[self.y[:] == 0, 0].numpy(),
                 self.x[self.y[:] == 0, 1].numpy(), 'o', label="y = 0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(),
                 self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y = 1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(),
                 self.x[self.y[:] == 2, 1].numpy(), 'go', label="y = 2")
        plt.legend()


# Create network model class
class Net(nn.Module):
    def __init__(self, _layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(_layers, _layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = F.relu(linear_transform(x))
            else:
                y = linear_transform(x)
        return y


# Define the function for training the model
def train(data_set, model, _criterion, train_loader, _optimizer, epochs=100):
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        for x, y in train_loader:
            _optimizer.zero_grad()
            yhat = model(x)
            loss = _criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        ACC.append(accuracy(model, data_set))

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color=color)
    ax1.set_xlabel('Iteration', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
    return LOSS


# The function to calculate the accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()


# Create a Dataset object
data_set = Data()
data_set.plot_stuff()
# data_set.y = data_set.y.view(-1), this line doesn't do anything
# The original y.shape is [3],
# if we set y.view(1, -1), y.shape will be changed to [1, 3]
data_set.y = data_set.y.view(-1)

# # Train the model with 1 hidden layer with 50 neurons
# Layers = [2, 50, 3]
# model = Net(Layers)
# learning_rate = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train_loader = DataLoader(dataset=data_set, batch_size=20)
# criterion = nn.CrossEntropyLoss()
# LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
#
# plot_decision_regions_3class(model, data_set)


# Train the model with 2 hidden layers with 20 neurons
Layers = [2, 10, 10, 3]
model = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=1000)

plot_decision_regions_3class(model, data_set)

# # Train the model with 3 hidden layers with 10 neurons
# Layers = [2, 10, 10, 10, 3]
# model = Net(Layers)
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train_loader = DataLoader(dataset=data_set, batch_size=20)
# criterion = nn.CrossEntropyLoss()
# LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=1000)
#
# plot_decision_regions_3class(model, data_set)
