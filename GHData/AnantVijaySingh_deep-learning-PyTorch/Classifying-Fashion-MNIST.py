import torch
import helper
import matplotlib.pyplot as plt
from torch import nn, optim
from collections import OrderedDict

# Download the MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [256, 128, 64]
output_size = 10

# Model
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
    ('relu3', nn.ReLU()),
    ('fc4', nn.Linear(hidden_sizes[2], output_size)),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))

# Define the loss
criterion = nn.NLLLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, lables in trainloader:
        # Flatten Image
        features = images.view(images.shape[0], -1)

        # Clear the gradients
        optimizer.zero_grad()

        # Calculate Output
        outputinlogvalue = model(features)

        # Calculate Loss
        loss = criterion(outputinlogvalue, lables)

        # Calculate gradient using autograd
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f'Training Loss: {running_loss/len(trainloader)}')


# Test model

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Output (probabilities)
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')