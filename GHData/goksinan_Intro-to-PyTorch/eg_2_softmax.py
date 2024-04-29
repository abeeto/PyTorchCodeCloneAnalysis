# MNIST example
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt


def activation(x):
    """Sigmoid activation function

    Arguments
    ---------
    x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))


def softmax(x):
    """Softmax function

    Arguments
    ---------
    x: torch.Tensor
    """
    den = torch.sum(torch.exp(x), dim=1)
    return torch.exp(x)/den.view(-1,1)


# Let's download the data
from torchvision import datasets, transforms
# Define a transfotm to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=False)])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
# Trainloader helps us to load data in batches during iterations
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Let's look at what out trainloader brings
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
# Plot a sample image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Define the size of each layer in out network
n_input = 784 # Image size is 28x28. We will transform this into a vector
n_hidden = 32
n_output = 10

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.matmul(inputs, W1) + B1)
out = activation(torch.matmul(h, W2) + B2)
print(out)

probabilities = softmax(out)

