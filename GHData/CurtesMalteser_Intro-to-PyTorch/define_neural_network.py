import numpy as np
import torch
from torch import nn

import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import OrderedDict

from network import Network

# Define a transform to normalize the data
# Subtracts 0.5 on normalize and divide by 0.5 to make the values range from -1 to 1
# because initial pixel values range are from 0 to 1 for each pixel
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
train_set = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Download and load the test data
test_set = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

model = Network()
print(model)

# Initialize weights and biases
print(model.fc1.weight)
print(model.fc1.bias)

# Re-init the bias of fc1
model.fc1.bias.data.fill_(0)
print(model.fc1.bias)

# Initialize weights with normal distribution of 0.1
model.fc1.weight.data.normal_(std=0.1)
print(model.fc1.weight)

# Pass data forward thought the network and display output
images, labels = next(iter(train_loader))

# Get batch size from tensor, which in this case is 64
# 784 is the 28*28 correspondent to img width and height
# and 1 layer since images are grayscale
batch_size_from_tensor = images.shape[0]
print(batch_size_from_tensor)
images.resize_(batch_size_from_tensor, 1, 784)

# probability distribution
ps = model.forward(images[0])

# Call view here covert image back to original size,
# is similar to resize, but return a tensor instead
helper.view_classify(images[0].view(1, 28, 28), ps)

# HyperParameters for network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Same as Network but with Sequential
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(hidden_sizes[1], output_size)),
    ('softmax', nn.Softmax(dim=1))]))

helper.pass_forward(train_loader, model)

## TODO: Your network here
hidden_sizes = [400, 200, 100]
output_size = 10

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
    ('relu3', nn.ReLU()),
    ('output', nn.Linear(hidden_sizes[2], output_size)),
    ('softmax', nn.Softmax(dim=1))]))

helper.pass_forward(train_loader, model)

data_transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(100),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])
