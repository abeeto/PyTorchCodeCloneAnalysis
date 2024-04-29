####################################################################
# Neural Network with multiple classes
####################################################################
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np


# Define a function to plot accuracy and loss
def plot_accuracy_loss(training_results):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()


# Define a function to plot model parameters
def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())


# Define a function to display data
def show_data(data_sample):
    plt.imshow(data_sample.numpy().reshape(28, 28), cmap='gray')
    plt.show()


# Define a Neural Network class
class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# Define a training function to train the model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs):
    i = 0
    results = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x.view(-1, 28*28))
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            results['training_loss'].append(loss.item())

        correct = 0
        for x, y in validation_loader:
            y_val = model(x.view(-1, 28*28))
            _, label = torch.max(y_val, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(validation_dataset))
        results['validation_accuracy'].append(accuracy)
    return results


# Create training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# Create validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# Create criterion function
criterion = nn.CrossEntropyLoss()
# Create data loader for both train dataset and valdiate dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Create the model with 100 neurons
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
model = Net(input_dim, hidden_dim, output_dim)

# Set the learning rate and the optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
training_results = train(model, criterion, train_loader,
                         validation_loader, optimizer, epochs=1)

# Plot the accuracy and loss
plot_accuracy_loss(training_results)

# Plot the first five misclassified samples
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data(x)
        count += 1
    if count >= 5:
        break

# Use nn.Sequential to build the same model.
model2 = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                             torch.nn.Sigmoid(),
                             torch.nn.Linear(hidden_dim, output_dim))
