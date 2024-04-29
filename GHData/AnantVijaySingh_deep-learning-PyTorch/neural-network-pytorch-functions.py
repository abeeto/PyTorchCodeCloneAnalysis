# Building NN using PyTorch functions

from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    # Here we're inheriting from nn.Module. Combined with super().__init__() this creates a class that tracks the
    # architecture and provides a lot of useful methods and attributes.
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation. This line creates a module for a linear transformation,
        # x W + b , with 784 inputs and 256 outputs and assigns it to self.hidden. The module automatically creates
        # the weight and bias tensors which we'll use in the forward method. You can access the weight and bias
        # tensors once the network (net) is created with net.hidden.weight and net.hidden.bias.
        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        #
        # We defined operations for the sigmoid activation and softmax output. Setting dim=1 in nn.Softmax(dim=1)
        # calculates softmax across the columns.
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward_simple_approach(self, x):  # Needs to be named 'forward' as it is supposed to override the parent class's function
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

    # You can define the network somewhat more concisely and clearly using the torch.nn.functional module. This is
    # the most common way you'll see networks defined as many operations are simple element-wise functions. We
    # normally import this module as F, import torch.nn.functional as F.

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x


# Create the network and look at it's text representation
model = Network()
print(model)

"""
----------------------------------
Importing and cleaning data for NN
----------------------------------
"""

# Import necessary packages
import torch
import matplotlib.pyplot as plt

# Download the MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# We have the training data loaded into trainloader and we make that an iterator with iter(trainloader). Later,
# we'll use this to loop through the dataset for training
#
# We created the trainloader with a batch size of 64, and shuffle=True. The batch size is the number of images we get
# in one iteration from the data loader and pass through our network, often called a batch. And shuffle=True tells it
# to shuffle the dataset every time we start going through the data loader again. But here I'm just grabbing the
# first batch so we can check out the data. We can see below that images is just a tensor with size (64, 1, 28,
# 28). So, 64 images per batch, 1 color channel, and 28x28 images.

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# Show test image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
plt.show()
