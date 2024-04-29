##########################################################
# Neural Network with only 1 layer
##########################################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
torch.manual_seed(0)


# The function for plotting the model
def PlotStuff(X, Y, model, epoch, leg=True):
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg:
        plt.legend()
    else:
        pass


# Define the class Net for Neural Network model
class Net(nn.Module):
    def __init__(self, input_size, node, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, node)
        self.linear2 = nn.Linear(node, output_size)
        # Define the first linear layer as an attribute, this is not good practice
        self.a1 = None
        self.l1 = None
        self.l2 = None

    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = torch.sigmoid(self.l1)
        self.l2 = self.linear2(self.a1)
        __yhat = torch.sigmoid(self.linear2(self.a1))
        # y_l1 = torch.sigmoid(self.linear1(x))
        # yhat = torch.sigmoid(self.linear2(y_l1))
        return __yhat


# Define the training function
def train(X, Y, model, optimizer, criterion, iter=1000):
    cost = []
    total = 0
    for i in range(iter):
        total = 0
        for x, y in zip(X, Y):
            yhat_ = model(x)
            loss = criterion(yhat_, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # cumulative loss
            total += loss.item()
        cost.append(total)
        if not i % 300:
            plt.figure()
            PlotStuff(X, Y, model, i, leg=True)
            plt.show()
            plt.figure()
            model(X)
            plt.scatter(model.a1.detach().numpy()[:, 0],
                        model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('activations')
            plt.show()
    return cost


# Make some data
X = torch.arange(-20., 20., 1).view(-1, 1)  # .type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4.) * (X[:, 0] < 4)] = 1.0


# Define Loss function
def criterion(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) +
                          (1 - labels) * torch.log(1 - outputs))
    return out


# Train the model
# size of input
input_size = 1
# size of the hidden layer
H = 2
# number of outputs
output_size = 1
# learning rate
lr = 0.1
# Create model
model = Net(input_size, H, output_size)
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr)
# train the model
cost_cross = train(X, Y, model, optimizer, criterion, iter=600)
print(cost_cross)

# Plot the loss
plt.figure()
plt.plot(cost_cross)
plt.xlabel('iter')
plt.ylabel('Cross entropy loss')
plt.show()

# Validation
x_test = torch.tensor([[-5.0], [2.0], [6.0]])
yhat = model(x_test)
print(yhat)
