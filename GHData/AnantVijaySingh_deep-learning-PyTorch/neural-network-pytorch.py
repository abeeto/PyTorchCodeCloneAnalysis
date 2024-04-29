# Import necessary packages
import matplotlib as matplotlib
import numpy as np
import torch
# import helper
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


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


# Our images are 28x28 2D tensors, so we need to convert them into 1D vectors. Thinking about sizes, we need to convert
# the batch of images with shape (64, 1, 28, 28) to a have a shape of (64, 784), 784 is 28 times 28. This is
# typically called flattening, we flattened the 2D images into 1D vectors.

features = images.view((64, 784))
print(features.shape)

# Another way to flatten the image is:
# Flatten the input images
features = images.view(images.shape[0], -1)  # -1 is used as a shorthand to automate the process. The function
# determines the shape required to ensure that all elements are captured. In this case it is 784

# Define the size of each layer in our network
n_input = features.shape[1]  # Number of input units, must match number of input features
n_hidden = 256  # Number of hidden units
n_output = 10  # Number of output units

# Initializing weight matrices
weights_input_hidden = torch.randn(n_input, n_hidden)
weights_hidden_output = torch.randn(n_hidden, n_output)

# Initializing bias terms for hidden and output layers
bias_input_hidden = torch.randn(1, n_hidden)
bias_hidden_output = torch.randn(1, n_output)

output_input_hidden = activation(torch.mm(features, weights_input_hidden) + bias_input_hidden)
output_hidden_output = torch.mm(output_input_hidden, weights_hidden_output) + bias_hidden_output

print(output_hidden_output.shape)
print(output_hidden_output)


# Now we have 10 outputs for our network. We want to pass in an image to our network and get out a probability
# distribution over the classes that tells us the likely class(es) the image belongs to.

# Implementing a function softmax that performs the softmax calculation and returns probability distributions for each
# example in the batch. Note that you'll need to pay attention to the shapes when doing this. If we have a tensor a
# with shape (64, 10) and a tensor b with shape (64,), doing a/b will give you an error because PyTorch will try to
# do the division across the columns (called broadcasting) and we'll get a size mismatch. The way to think about
# this is for each of the 64 examples, we only want to divide by one value, the sum in the denominator. So we need
# b to have a shape of (64, 1). This way PyTorch will divide the 10 values in each row of a by the one value in each
# row of b. Pay attention to how we take the sum as well. We'll need to define the dim keyword in torch.sum.
# Setting dim=0 takes the sum across the rows while dim=1 takes the sum across the columns.

def softmax(x):
    print(x.shape)
    print(torch.sum(torch.exp(x), dim=1).shape)  # Row metarix
    print(torch.sum(torch.exp(x), dim=1).view(-1, 1).shape)  # Column matrix
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(output_hidden_output)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))
