from collections import OrderedDict

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import helper

# PyTorch will track operations on this tensor using auto grad
x = torch.randn(2, 2, requires_grad=True)
print(x)

y = x ** 2
print(y)

# grad_fn shows the function that generated this variable
print(y.grad_fn)

# Reduce the tensor y to a scalar value, the mean
z = y.mean()
print(z)

# Will print None because the backward pass through the previous operations wasn't done yet
print(x.grad)

# Calculate the gradient for z with respect to x
z.backward()
print(x.grad)
print(x / 2)

device = torch.cuda.current_device()

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
train_set = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# HyperParameters for network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('logits', nn.Linear(hidden_sizes[1], output_size))])).cuda()

# Training the network
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print('Before', model.fc1.weight)

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

images.resize_(64, input_size)


# Zero out all gradients that are set on Tensors
# It's needed to avoid accumulated values
optimizer.zero_grad()

# Forward pass
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model.fc1.weight.grad)
optimizer.step()

print('Updated weights - ', model.fc1.weight)

optimizer = optim.SGD(model.parameters(), lr=0.003)

# Train the model
epochs = 3
print_every = 40
steps = 0
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(train_loader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        # Forward and backward passes
        output = model.forward(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e + 1, epochs),
                  "Loss: {:.4f}".format(running_loss / print_every))

            running_loss = 0

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

# check out trained network predictions.
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)

helper.view_classify(img.view(1, 28, 28).cpu(), ps.cpu())
