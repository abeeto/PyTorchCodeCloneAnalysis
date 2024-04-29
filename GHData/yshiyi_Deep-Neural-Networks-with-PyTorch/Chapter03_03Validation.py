##########################################################
# Linear regression: Training and Validation Data
##########################################################
# Import libraries we need for this lab, and set the random seed
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# Create dataset class
class dataset(Dataset):
    def __init__(self, train=True):
        self.x = torch.arange(-3., 3., 0.1).view(-1, 1)
        self.f = -3 * self.x + 1
        self.y = self.f + torch.randn(self.x.size())
        self.len = self.x.shape[0]
        if train:
            self.y[0] = 0
            self.y[50:55] = 20

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Create linear regression class
class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat

# Create training dataset
train_data = dataset()
# Create validation dataset
val_data = dataset(train=False)

# Plot data points
plt.plot(train_data.x.numpy(), train_data.y.numpy(), 'xr', label="training data ")
plt.plot(train_data.x.numpy(), train_data.f.numpy(), label="true function  ")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Create MSELoss function and DataLoader
criterion = nn.MSELoss()  # torch.nn.MSELoss()
trainloader = DataLoader(dataset=train_data, batch_size=1)
valloader = DataLoader(dataset=val_data, batch_size=1)

# Create Learning Rate list, the error lists and the MODELS list
learning_rates = [0.0001, 0.001, 0.01, 0.1]
train_error = torch.zeros(len(learning_rates))
val_error = torch.zeros(len(learning_rates))
model_list = []

# Define the train model function and train the model
def train_model(iter):
    for i, lr in enumerate(learning_rates):
        model = linear_regression(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        for epoch in range(iter):
            for x, y in trainloader:
                yhat = model(x)
                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Check loss after training process finishes
        # Check loss of training data
        Yhat = model(train_data.x)
        loss_train = criterion(Yhat, train_data.y)
        train_error[i] = loss_train.item()

        # Check loss of validation data
        Yhat = model(val_data.x)
        loss_val = criterion(Yhat, val_data.y)
        val_error[i] = loss_val.item()
        model_list.append(model)


train_model(10)
# Plot the training loss and validation loss
plt.semilogx(learning_rates, train_error.numpy(), label='training loss/total Loss')
plt.semilogx(learning_rates, val_error.numpy(), label='validation cost/total Loss')
plt.ylabel('Cost\ Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()

# Plot the predictions
i = 0
for model, learning_rate in zip(model_list, learning_rates):
    yhat = model(val_data.x)
    plt.plot(val_data.x.numpy(), yhat.detach().numpy(), label='lr:' + str(learning_rate))
    # print('i', yhat.detach().numpy()[0:3])
plt.plot(val_data.x.numpy(), val_data.f.numpy(), 'or', label='validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
