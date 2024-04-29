# part 2: neural networks in pytorch


import numpy as np
import torch
from torchvision import datasets, transforms
import helper

import matplotlib.pyplot as plt
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
# print(type(images))
# print(images.shape)
# print(labels.shape)

# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
# plt.show()

# build a simple network for this dataset using weight matrices and matrix

'''
## Solution
def activation(x):
    return 1/(1+torch.exp(-x))

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
# print(len(out))


def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# should have the right shape
print(probabilities.shape)

# does it sum to 1
print(probabilities)
'''


from torch import nn

# use nn to build network
'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        # ~~this line creates module for linear transformation, xW+b, with 784 inputs
        #   and 256 outputs, assigning it to self.hidden
        # module automatically creates weight and bias tensors to be used in ~forward~
        #   method. They can be accessed once network is created with ~net.hidden.weight~
        #   and ~net.hidden.bias~
        self.hidden = nn.Linear(784, 256)
        
        # Output layer, 10 units - one for each digit
        # ~~same for this line as above
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        # ~~setting dim=1 calculates softmax across columns, =0 across rows
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Pass input tensor through each of the operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = Network()
print(model)
'''

# ~~or use more concise definition with torch.nn.functional module imported as F
#   .this is the most common wat
import torch.nn.functional as F

'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)

        return x


model = Network()
print(model)
'''

'''
    - Other functions can be used as activation fns.
    - Only requirement is that to appx non-linear fn, 
        atn-fns must be  non-linear.
    - Other activation fns, Tanh, ReLU (rectified linear unit)
    - ReLU is used almost exclusively as atn-fn for hidden layers
'''

# Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation,
# then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation

# good practice to name layers by type of nework, fc to represent fully-connected layer for example.

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x = self.fc1(x)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.softmax(self.fc3(x), dim=1)

        return x

# model = Network()
# print(model)
# print(model.fc1.weight)
# print(model.fc1.bias)
'''
model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.01)
print(model.fc1.weight)
print(model.fc1.bias)
'''

'''
images.resize_(64, 1, 784)
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
'''

# build network by passing tensor sequentially through operations using nn.Sequential
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.Softmax(dim=1)
)

# print(model)
# images.resize_(images.shape[0], 1, 784)
# ps = model.forward(images[0,:])
# helper.view_classify(images[0].view(1,28,28), ps)

# OrderedDict can also be passed to name individual layers and operations,
# rather than using incremental integers
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
print(model)
print(model[0])
print(model.fc1)