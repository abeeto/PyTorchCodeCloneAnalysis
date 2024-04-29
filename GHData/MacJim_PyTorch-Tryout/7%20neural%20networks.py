# Source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# MARK: Network definition
# `nn.Module` represents a neural network.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 3x3 square convolution kernel size.
        # Default stride is 1.
        # Default padding is 0.
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # An affine (linear) operation, y = Wx + b
        # "fc" is short for "full connection".
        # Input sample size, output sample size.
        # Bias is enabled by default.
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window.
        # Max pooling is used for downsampling.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # You may specify a single number if the size is a square.
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]    # TODO: all dimensions except the batch dimension???
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


# MARK: Test the network
net = Net()
print(net)

# Obtain learnable parameters.
parameters = list(net.parameters())
print("Number of parameters:", len(parameters))
# Prints all operations in `forward`.
# for aParameter in parameters:
#     print(aParameter.size())

# Try a random 32x32 input.
# This input is a 4D tensor, representing: sample count (1), channel count (1), height, width. The first 2 parameters are required by `nn.Conv2d`.
input = torch.randn(1, 1, 32, 32)
output = net(input)
print("Output:", output)

# Zero the gradient buffers of all parameters and backprops with random gradients.
net.zero_grad()
output.backward(torch.randn(1, 10), retain_graph=True)


# MARK: Compute loss
target = torch.randn(10)    # Dummy target value used for this test.
target = target.view(1, -1)    # Make it the same shape as the output.
# Mean squared error (very common).
criterion = nn.MSELoss()

loss = criterion(output, target)
print("Loss:", loss)

# Call `loss.backward()` to differentiate wrt the loss.
print(loss.grad_fn)    # MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])    # ReLU


# MARK: Backprop
# Before backpropagating, clear the existing gradients. Otherwise gradients will be accumulated to existing gradients.
net.zero_grad()
print("conv1.bias.grad before backward():", net.conv1.bias.grad)
loss.backward(retain_graph=True)
print("conv1.bias.grad after backward():", net.conv1.bias.grad)


# MARK: Update the weights
# Using SGD in the legacy way.
# learningRate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learningRate)

# Using PyTorch's methods
optimizer = optim.SGD(net.parameters(), lr=0.01)

# TODO: In our training loop
def train():
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
