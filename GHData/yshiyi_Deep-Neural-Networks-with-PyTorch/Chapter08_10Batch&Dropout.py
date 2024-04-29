####################################################################
# Test Batch Normalization and Dropout
####################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
torch.manual_seed(0)


# Define a function for plot the decision region
def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X=data_set.x.numpy()
    y=data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label='y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label='y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label='y=2')
    plt.title("decision region")
    plt.legend()
    plt.show()


# Create the dataset class
class Data(Dataset):
    # Constructor
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + \
                np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j

        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the diagram
    def plot_data(self):
        plt.figure()
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'bo', label="y=0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y=1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'go', label="y=2")
        plt.legend()
        plt.show()


# Create neural network with Batch Normalization and Dropout
class Net_Adv(nn.Module):
    def __init__(self, layers, p=0.5):
        super(Net_Adv, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
            bn = nn.BatchNorm1d(output_size)
            self.bn.append(bn)

    def forward(self, x):
        L = len(self.hidden)
        for (l, LT, BN) in zip(range(L), self.hidden, self.bn):
            if l < L - 1:
                # x = BN(torch.relu(self.drop(LT(x))))  # BN&Dropout
                # x = BN(torch.relu(LT(x)))  # BN
                x = torch.relu(self.drop(LT(x)))  # Dropout
                # x = torch.relu(LT(x))  # normal
                # x = torch.sigmoid(LT(x))  # sigmoid
                # x = torch.tanh(LT(x))  # tanh
            else:
                yhat = LT(x)
        return yhat


# Create general neural network with ModuleList()
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                yhat = linear_transform(x)
        return yhat


# Define the function for training the model
def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    results = {'LOSS': [], 'ACC': []}
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results['LOSS'].append(criterion(model(data_set.x), data_set.y))
        results['ACC'].append(accuracy(model, data_set))
    return results


# Define a function for calculating accuracy
def accuracy(model, data_set):
    _, label = torch.max(model(data_set.x), 1)
    return (label == data_set.y).numpy().mean()


# Create the dataset and plot it
data_set = Data()
data_set.plot_data()
data_set.y = data_set.y.view(-1)


# Train a model with general neural network without momemtum
Layers = [2, 100, 3]
model = Net(Layers)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
result_norm = train(data_set, model, criterion,
                    train_loader, optimizer, epochs=100)
plot_decision_regions_3class(model, data_set)

# Train a model with advanced neural network with momentum=0.4
Layers = [2, 100, 3]
model = Net_Adv(Layers, p=0.0)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.4)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
result_adv = train(data_set, model, criterion,
                   train_loader, optimizer, epochs=100)
plot_decision_regions_3class(model, data_set)

# Plot loss
plt.figure()
plt.plot(result_norm['LOSS'], label='Norm')
plt.plot(result_adv['LOSS'], label='Adv')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Total Loss')
plt.show()

# Plot accuracy
plt.figure()
plt.plot(result_norm['ACC'], label='Norm')
plt.plot(result_adv['ACC'], label='Adv')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()

