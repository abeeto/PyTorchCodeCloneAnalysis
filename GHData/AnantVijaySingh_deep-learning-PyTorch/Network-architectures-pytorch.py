import torch
from torch import nn
import matplotlib.pyplot as plt
import helper
from collections import OrderedDict
from torch import optim

"""

Downloading and preparing data for training and testing

"""
# Download the MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Grab some data
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
features = images.view(images.shape[0], -1)

# Hyperparameters for our network
input_size = features.shape[1]
hidden_sizes = [128, 64]
output_size = 10

"""

Defining Neural Network, Loss Calculations, Gradient Decent

"""

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(hidden_sizes[1], output_size)),
    ('LogSoftMax', nn.LogSoftmax(dim=1))
]))

# Define the loss
criterion = nn.NLLLoss()

# Optimizer we'll use to update the weights with the gradients. Requires the parameters to optimize and a learning rate.
optimizer = optim.SGD(model.parameters(), lr=0.003)

# Steps are replicated below in the training loop
# # Forward pass, get our log-probabilities
# logps = model(features)
#
# # Calculate the loss with the logps and the labels
# loss = criterion(logps, labels)


"""

Training the Neural Network using 

"""

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        features = images.view(images.shape[0], -1)

        # CRITICAL: Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # TODO: Training pass

        # Model Output
        logps = model(features)

        # Loss
        loss = criterion(logps, labels)

        # Calculate gradient using autograde
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")



"""

Checking out the trained networks predictions.

"""

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities as we used LogSoftMax instead of simple softmax, need to take
# exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)