#########################################################################
# Initialization Weights
# In this script, we will see the problem with initializing the weights
# with the same values.
#########################################################################
import torch
import torch.nn as nn
from torch import sigmoid
import matplotlib.pyplot as plt
import sys
torch.manual_seed(0)


# The function for plotting the model
def PlotStuff(X, Y, model, epoch, leg=True):
    plt.figure()
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg:
        plt.legend()
    else:
        pass


# Define the class Net
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

        # # Set weights to zero
        # self.linear1.weight.data.zero_()
        # self.linear2.weight.data.zero_()
        # # Set bias to one
        # self.linear1.bias.data.uniform_(1.0, 1.0)
        # self.linear2.bias.data.uniform_(1.0, 1.0)

        self.a1 = None

    def forward(self, x):
        self.a1 = sigmoid(self.linear1(x))
        yhat = sigmoid(self.linear2(self.a1))
        return yhat


# Define the training function
def train(X, Y, model, optimizer, criterion, epochs=1000):
    cost = []
    for epoch in range(epochs):
        total = 0
        for x, y in zip(X, Y):
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total += loss.item()

        cost.append(total)
        if epoch % 300 == 0:
            PlotStuff(X, Y, model, epoch, leg=True)
            plt.show()
            model(X)
            plt.figure()
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('activations')
            plt.show()
    return cost


# Make some data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)  # X.size([_, 1])
Y = torch.zeros(X.shape[0])  # Y.size([_])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0  # X[:, 0].size([_])

# Define loss function
def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) +
                          (1 - labels) * torch.log(1 - outputs))
    return out

# Train the model
D_in = 1
H = 2
D_out = 1
learning_rate = 0.1
model = Net(D_in, H, D_out)
print(model.state_dict())

# Set the all weights to one and all bias to zero
model.state_dict()['linear1.weight'][0] = 1.0
model.state_dict()['linear1.weight'][1] = 1.0
model.state_dict()['linear1.bias'][0] = 0.0
model.state_dict()['linear1.bias'][1] = 0.0
model.state_dict()['linear2.weight'][0] = 1.0
model.state_dict()['linear2.bias'][0] = 0.0
print(model.state_dict())

# # Define optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# # train the model
# cost_cross = train(X, Y, model, optimizer, criterion_cross)
#
# # Plot the lost
# plt.figure()
# plt.plot(cost_cross)
# plt.xlabel('epoch')
# plt.title('cross entropy loss')
# plt.show()
#
# print(model.state_dict())
# # validation
# yhat = model(torch.tensor([[-2.0], [0.0], [2.0]]))
# print(yhat)


# Using the MSE cost
model2 = Net(D_in, H, D_out)
# print(model.state_dict())
# sys.exit(0)
optimizer_mse = torch.optim.SGD(model2.parameters(), lr=learning_rate)
criterion_mse = nn.MSELoss()
cost_mse = train(X, Y, model2, optimizer_mse, criterion_mse)

# Plot the lost
plt.figure()
plt.plot(cost_mse)
plt.xlabel('epoch')
plt.title('MSE loss')
plt.show()


