####################################################################
# Neural Network with only 1 layer and multiple hidden neutrons
# Using nn.Sequential to create the network
####################################################################
import torch
import torch.nn as nn
from torch import sigmoid, tanh, relu
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt
import torch.nn.functional as F
import sys
torch.manual_seed(0)


# Define the plotting functions
def get_hist(model, data_set):
    activations = model.activation(data_set.x)
    for i, activation in enumerate(activations):
        plt.figure()
        plt.hist(activation.numpy(), 4, density=True)
        plt.title("Activation layer " + str(i+1))
        plt.xlabel("Activation")
        plt.xlabel("Activation")
        plt.legend()
        plt.show()


def PlotStuff(X, Y, model=None, leg=False):
    plt.figure()
    plt.plot(X[Y == 0].numpy(), Y[Y == 0].numpy(), 'or', label='training points y=0 ')
    plt.plot(X[Y == 1].numpy(), Y[Y == 1].numpy(), 'ob', label='training points y=1 ')

    if model != None:
        plt.plot(X.numpy(), model(X).detach().numpy(), label='neral network ')

    plt.legend()
    plt.show()


# Create Data class
class Data(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0] > -10) * (self.x[:, 0] < -5)] = 1
        self.y[(self.x[:, 0] > 5) * (self.x[:, 0] < 10)] = 1
        # self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Create network class
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y_out = sigmoid(self.linear1(x))
        yhat = sigmoid(self.linear2(y_out))
        return yhat


# Create data
data_set = Data()
PlotStuff(data_set.x, data_set.y, leg=False)
# Create model
model = Net(1, 7, 1)
# Create criterion
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# Create data loader
train_loader = DataLoader(dataset=data_set, batch_size=100)
# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# Create a function to train model
def train(data_set, model, criterion, train_loader, optimizer, iter=5, plot_num=10):
    cost = []
    for i in range(iter):
        total = 0
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()

        if i % plot_num == 1:
            PlotStuff(data_set.x, data_set.y, model)
        cost.append(total)
    return cost


# Train model
cost = train(data_set, model, criterion, train_loader, optimizer, 600, 300)
plt.figure()
plt.plot(cost)
plt.xlabel('iter')
plt.ylabel('Cross entropy loss')
plt.show()


# # Validation
# x_test = torch.tensor([[-6.0], [2.0], [6.0]])
# yhat = model(x_test)
# print(yhat)

# sys.exit(0)

####################################################################
# Using nn.Sequential to create the network
####################################################################
torch.manual_seed(0)
model2 = torch.nn.Sequential(
    torch.nn.Linear(1, 13),
    torch.nn.Sigmoid(),
    torch.nn.Linear(13, 1),
    torch.nn.Sigmoid()
)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)
cost2 = train(data_set, model2, criterion, train_loader, optimizer2, 600, 300)
plt.figure()
plt.plot(cost2)
plt.xlabel('iter')
plt.ylabel('Cross entropy loss')
plt.show()

# Validation
x_test = torch.tensor([[-6.0], [2.0], [6.0]])
yhat1 = model(x_test)
yhat2 = model2(x_test)
print(yhat1, yhat2)
