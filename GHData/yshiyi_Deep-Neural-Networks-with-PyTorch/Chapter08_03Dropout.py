####################################################################
# Multiple Hidden Layer Deep Network
# Using Dropout for Classification
# Used mostly when the number of data is not large
####################################################################
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import ListedColormap
import sys


# The function for plotting the diagram
def plot_decision_regions_3class(data_set, model=None):
    cmap_light = ListedColormap(['#0000FF', '#FF0000'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]

    Z = data_set.multi_dim_poly(newdata).flatten()
    f = np.zeros(Z.shape)
    f[Z > 0] = 1
    f = f.reshape(xx.shape)
    if model is not None:
        model.eval()
        XX = torch.Tensor(newdata)
        _, yhat = torch.max(model(XX), 1)
        yhat = yhat.numpy().reshape(xx.shape)
        plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
    else:
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.pcolormesh(xx, yy, f, cmap=cmap_light)

    plt.title("decision region vs True decision boundary")


# The function for calculating accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()


# Create data class for creating dataset object
class Data(Dataset):
    def __init__(self, N_SAMPLES=1000, noise_std=0.15, _train=True):
        a = np.matrix([-1, 1, 2, 1, 1, -3, 1]).T
        self.x = np.matrix(np.random.rand(N_SAMPLES, 2))
        self.f = np.array(a[0] + self.x * a[1:3] +
                          np.multiply(self.x[:, 0], self.x[:, 1]) * a[4] +
                          np.multiply(self.x, self.x) * a[5:7]).flatten()
        self.a = a

        self.y = np.zeros(N_SAMPLES)
        self.y[self.f > 0] = 1
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)
        self.x = self.x + noise_std * torch.randn(self.x.size())
        self.f = torch.from_numpy(self.f)
        self.len = self.x.shape[0]
        if _train:
            torch.manual_seed(1)
            self.x = self.x + noise_std * torch.randn(self.x.size())
            torch.manual_seed(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    # Plot the diagram
    def plot(self):
        X = data_set.x.numpy()
        y = data_set.y.numpy()
        h = .02
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = data_set.multi_dim_poly(np.c_[xx.ravel(), yy.ravel()]).flatten()
        f = np.zeros(Z.shape)
        f[Z > 0] = 1
        f = f.reshape(xx.shape)

        plt.title('True decision boundary and sample points with noise ')
        plt.plot(self.x[self.y == 0, 0].numpy(), self.x[self.y == 0, 1].numpy(),
                 'bo', label='y=0')
        plt.plot(self.x[self.y == 1, 0].numpy(), self.x[self.y == 1, 1].numpy(),
                 'ro', label='y=1')
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()

    # Make a multidimension ploynomial function
    def multi_dim_poly(self, x):
        x = np.matrix(x)
        out = np.array(self.a[0] + x * self.a[1:3] +
                       np.multiply(x[:, 0], x[:, 1]) * self.a[4] +
                       np.multiply(x, x) * self.a[5:7])
        out = np.array(out)
        return out


# Create a dataset object
data_set = Data(noise_std=0.2)
# sys.exit(0)
data_set.plot()
# sys.exit(0)

# Get some validation data
torch.manual_seed(0)
validation_set = Data(_train=False)


# Create Net class
class Net(nn.Module):
    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)

    def forward(self, x):
        x = F.relu(self.drop(self.linear1(x)))
        x = F.relu(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x


# Create two model objects: model without dropout and model with dropout
model = Net(2, 300, 2)
model_drop = Net(2, 300, 2, p=0.5)

# Set the model to training mode
model_drop.train()

# Set optimizer functions and criterion functions
optimizer_overfit = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Initialize the LOSS dictionary to store the loss
LOSS = {}
LOSS['training data no dropout'] = []
LOSS['validation data no dropout'] = []
LOSS['training data dropout'] = []
LOSS['validation data dropout'] = []


# Train the model
def train(epochs):
    for epoch in range(epochs):
        yhat = model(data_set.x)
        yhat_drop = model_drop(data_set.x)
        loss = criterion(yhat, data_set.y)
        loss_drop = criterion(yhat_drop, data_set.y)

        # store the loss for both the training and validation data for both models
        LOSS['training data no dropout'].append(loss.item())
        LOSS['training data dropout'].append(loss_drop.item())
        model_drop.eval()
        LOSS['validation data no dropout'].append(criterion(model(validation_set.x), validation_set.y).item())
        LOSS['validation data dropout'].append(criterion(model_drop(validation_set.x), validation_set.y).item())
        model_drop.train()

        optimizer_overfit.zero_grad()
        optimizer_drop.zero_grad()
        loss.backward()
        loss_drop.backward()
        optimizer_overfit.step()
        optimizer_drop.step()


train(500)

# Set the model to evaluation model
model_drop.eval()

# Print out the accuracy of the model without dropout
print("The accuracy of the model without dropout: ",
      accuracy(model, validation_set))

# Print out the accuracy of the model without dropout
print("The accuracy of the model with dropout: ",
      accuracy(model_drop, validation_set))

plt.figure()
plt.plot(LOSS['training data no dropout'], label='T-no dropout')
plt.plot(LOSS['training data dropout'], label='T-dropout')
plt.legend()
plt.show()

plt.figure()
plt.plot(LOSS['validation data no dropout'], label='V-no dropout')
plt.plot(LOSS['validation data dropout'], label='V-dropout')
plt.legend()
plt.show()

# Plot the loss
plt.figure(figsize=(6.1, 8))
for key, value in LOSS.items():
    plt.plot(np.log(np.array(value)), label=key)
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("Log of cost or total loss")
