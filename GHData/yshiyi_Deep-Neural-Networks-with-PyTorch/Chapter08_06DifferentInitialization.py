#########################################################################
# Test different initialization methods:
# He, Uniform, Default and Xavier Uniform
#########################################################################
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)


# Define the class for neural network model with He Initialization
class Net_He(nn.Module):
    def __init__(self, Layers):
        super(Net_He, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# Define the neural network with Xavier initialization
class Net_Xavier(nn.Module):
    def __init__(self, Layers):
        super(Net_Xavier, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                yhat = linear_transform(x)
        return yhat


# Define the neural network with Uniform initialization
class Net_uniform(nn.Module):
    def __init__(self, Layers):
        super(Net_uniform, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            # torch.nn.init.uniform_(linear.weight, 0, 1)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for l, linear_transform in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                yhat = linear_transform(x)
        return yhat


# Define the neural network with Pytorch Default initialization
class Net_default(nn.Module):
    def __init__(self, Layers):
        super(Net_default, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for l, linear_transform in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                yhat = linear_transform(x)
        return yhat


# Define function to train model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    loss_accuracy = {'training_loss': [], 'validation_loss': []}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            yhat = model(x.view(-1, 28 * 28))
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_accuracy['training_loss'].append(loss.item())

        correct = 0
        for x, y in validation_loader:
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(validation_dataset))
        loss_accuracy['validation_loss'].append(accuracy)

    return loss_accuracy


# Create the train dataset
train_dataset = dsets.MNIST(root='./data', train=True,
                            download=True, transform=transforms.ToTensor())
# Create the validation dataset
validation_dataset = dsets.MNIST(root='./data', train=False,
                                 download=True, transform=transforms.ToTensor())
# Create DataLoader for both train dataset and validation dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=2000,
                               shuffle=False)

# Define criterion function
criterion = nn.CrossEntropyLoss()

# Set the parameters
input_dim = 28 * 28
output_dim = 10
# layers = [input_dim, 100, 10, 100, 10, 100, output_dim]
layers = [input_dim, 100, 200, 100, output_dim]
epochs = 15

# Train the model with default initialization
model = Net_default(layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader,
                         optimizer, epochs=epochs)

# Train the model with He initialization
model = Net_He(layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader,
                         optimizer, epochs=epochs)

# # Train the model with Xavier initialization
# model_Xavier = Net_Xavier(layers)
# optimizer = torch.optim.SGD(model_Xavier.parameters(), lr=learning_rate)
# training_results_Xavier = train(model_Xavier, criterion, train_loader,
#                                 validation_loader, optimizer, epochs=epochs)
#
# # Train the model with Uniform initialization
# model_Uniform = Net_uniform(layers)
# optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
# training_results_Uniform = train(model_Uniform, criterion, train_loader,
#                                  validation_loader, optimizer, epochs=epochs)

# Plot the loss
plt.figure()
plt.plot(training_results_Xavier['training_loss'], label='Xavier')
plt.plot(training_results['training_loss'], label='Default')
plt.plot(training_results_Uniform['training_loss'], label='Uniform')
plt.ylabel('loss')
plt.xlabel('iteration ')
plt.title('training loss iterations')
plt.legend()
plt.show()

# Plot the accuracy
plt.figure()
plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier')
plt.plot(training_results['validation_accuracy'], label='Default')
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform')
plt.ylabel('validation accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()
