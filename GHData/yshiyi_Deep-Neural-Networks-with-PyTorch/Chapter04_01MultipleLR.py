##########################################################
# Multiple Linear Regression:
# Training multiple weighting parameters and one bias
##########################################################
import torch
import sys
torch.manual_seed(1)

# Set the weight and bias
# Note: we create w as a column tensor/vector
# w = torch.tensor([2.0, 3.0], requires_grad=True).view(-1, 1)
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)


# Define Prediction Function
def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat


# Calculate yhat
x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)

# Sample tensor X
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
# Make the prediction of X
yhat = forward(X)
print("The result: ", yhat)

##########################################################
# Class Linear
##########################################################
# Make a linear regression model using build-in function
# Initial values of parameters are random
model = torch.nn.Linear(2, 1)
# print(model.state_dict().keys())

# Make a prediction of x
yhat = model(x)
print("The result: ", yhat)
# Note: if
# yhat1 = torch.mm(x, model.state_dict()["weight"].view(-1, 1))+model.state_dict()["bias"]
# print("The result2: ", yhat1)
# sys.exit(0)
# Make a prediction of X
yhat = model(X)
print("The result: ", yhat)


##########################################################
# Build Custom Modules
##########################################################
# Create linear_regression Class
class LR(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, __x):
        __yhat = self.linear(__x)
        return __yhat


model = LR(2, 1)
# Print model parameters
print("The parameters: ", list(model.parameters()))
# Print model parameters
print("The parameters: ", model.state_dict())
print(model.state_dict()["linear.weight"][0][0].item())
# Make a prediction of x
yhat = model(x)
print("The result: ", yhat)
# Make a prediction of X
yhat = model(X)
print("The result: ", yhat)


##########################################################
# Train multiple parameters
##########################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)


# The function for plotting 2D
def Plot_2D_Plane(model, dataset, n=0):
    w1 = model.state_dict()['linear.weight'].numpy()[0][0]
    w2 = model.state_dict()['linear.weight'].numpy()[0][1]
    b = model.state_dict()['linear.bias'].numpy()

    # Data
    x1 = data_set.x[:, 0].view(-1, 1).numpy()
    x2 = data_set.x[:, 1].view(-1, 1).numpy()
    y = data_set.y.numpy()

    # Make plane
    X1, X2 = np.meshgrid(np.arange(x1.min(), x1.max(), 0.05), np.arange(x2.min(), x2.max(), 0.05))
    yhat = w1 * X1 + w2 * X2 + b

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x1[:, 0], x2[:, 0], y[:, 0], 'ro', label='y')  # Scatter plot

    ax.plot_surface(X1, X2, yhat)  # Plane plot

    ax.set_xlabel('x1 ')
    ax.set_ylabel('x2 ')
    ax.set_zlabel('y')
    plt.title('estimated plane iteration:' + str(n))
    ax.legend()

    plt.show()


# Create a 2D dataset
class Dataset2D(Dataset):
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1.0
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f + torch.randn(self.f.size())
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Create a dataset object
data_set = Dataset2D()
# Create a linear regression model
model = LR(2, 1)
# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# Create the cost function
criterion = torch.nn.MSELoss()
# Create the data loader
data_loader = DataLoader(dataset=data_set, batch_size=2)

# Train the model
LOSS = []
print("Before training: ")
Plot_2D_Plane(model, data_set)


def train_model(iter):
    for i in range(iter):
        for x, y in data_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()


train_model(100)
print("After training:")
Plot_2D_Plane(model, data_set, iter)

plt.plot(LOSS)
plt.xlabel("Iterations ")
plt.ylabel("Cost/total loss ")
plt.show()
