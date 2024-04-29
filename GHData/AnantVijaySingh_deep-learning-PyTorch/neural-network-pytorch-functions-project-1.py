# Project: Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden
# layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above.
# You can use a ReLU activation with the nn.ReLU module or F.relu function.

from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

model = Network()
print(model)


# Initializing weights and biases
#
# The weights and such are automatically initialized for you, but it's possible to customize how they are
# initialized. The weights and biases are tensors attached to the layer you defined, you can get them with
# model.fc1.weight for instance.

print(model.fc1.weight)
print(model.fc1.bias)

# For custom initialization, we want to modify these tensors in place. These are actually autograd Variables,
# so we need to get back the actual tensors with model.fc1.weight.data. Once we have the tensors, we can fill them
# with zeros (for biases) or random normal values.

# Set biases to all zeros
model.fc1.bias.data.fill_(0)

# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)



"""
----------------------------------
Importing and cleaning data for NN
----------------------------------
"""


import torch
import matplotlib.pyplot as plt
import helper

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


# Show test image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
plt.show()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

"""
----------------------------------
Forward Propagation
----------------------------------
"""


# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx, :])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)


"""
----------------------------------
Using Sequential to build networks.
----------------------------------
"""


# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)



# The operations are availble by passing in the appropriate index. For example, if you want to get first Linear
# operation and look at the weights, you'd use model[0].

print(model[0])
model[0].weight

# You can also pass in an OrderedDict to name the individual layers and operations, instead of using incremental
# integers. Note that dictionary keys must be unique, so each operation must have a different name.

from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
print(model)

# Now you can access layers either by integer or the name

print(model[0])
print(model.fc1)

