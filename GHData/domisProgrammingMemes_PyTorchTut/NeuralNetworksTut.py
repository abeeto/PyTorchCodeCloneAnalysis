# third Tutorial - Neural Networks: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
# nn depends on autograd to definde models and differentiate them
# as nn.Module contains layers, and a method 'forward(input) that returns the output
# picture with convolutional nn which classiefies digit images!! note: picture is not 100% right
# Formula: Wout = ((Win + 2xPadding) - dilation x ((kernel_size -1) - 1) / (Stride + 1
# Formula(Copy): Hout = ⌊(Hin + 2 ×  padding[0] −  dilation[0] × ( kernel_size[0] − 1) − 1)/( stride[0]) +  1
# = (32 + 2*0 - 1*(3-1)-1)/1 + 1
# = (32 - 2 - 1) / 1) + 1
# = (29/1) + 1 = 30 --> daher müsste es im bild 5x5 conv sein!


# typical training procedure for a nn is:
# - define the nn that has some learnable parameters (or weights)
# - iterate over a dataset of inputs
# - process input through network
# - computate loss (how far is the output from being correct)
# - propagate gradients back into the network's parameters
# - update the weights of the network, typically using a simple update rule: weight = weight - learningr_ate * gradient

import torch
# use torch.nn for neural networks and torch.nn.functional for functions!
import torch.nn as nn
import torch.nn.functional as F

# lets define a network: (always as class!)
print("Define the Network")
class Net(nn.Module):

    # always need the init with super!
    def __init__(self):
        super(Net, self).__init__()
        # kernel
        # 1 input image channel, 6 output channels, 3x3 square convolution (3 is the filter which typically is 3 or 5)
        self.conv1 = nn.Conv2d(1, 6, 3)
        # first of conv2 has to be last of conv1!
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6 , 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) window
        # print(x.size(), "this is the size!!!!!")
        # x = self.conv1(x)
        # print(x.size(), "this is the size!!!!!")
        # x = F.relu(x)
        # print(x.size(), "this is the size!!!!!")
        # x = F.max_pool2d(x, 2)
        # print(x.size(), "this is the size!!!!!")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(x.size())
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# create a net with the defined params and method for the forward pass
net = Net()
print(net)

# just have to define forward function as backward function is automatically defined if using autograd!

# learnable params are returnded by net.parameters()
params = list(net.parameters())
print(len(params), "this is lenght of params")
print(params[0].size(), "those are conv1's weights") # conv1's .weights!

# expected size of the input of this net (LeNet) is 32x32. To use this net on the MNIST dataset please resize the images to 32x32
print("let's try random input")
input = torch.randn(1, 1, 32, 32)
print(input, "this is the input")
out = net(input)
print(out, "this is out")

# zero the gradient buffers of all params and backprop with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# notes: torch.nn only supports mini-batches. the entire torch.nn package only supports inputs that are a mini-batch of samples and not a single sample
# for example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
# if you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension

# recap:
# - torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor
# - nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
# - nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module
# - autograd.Function - Implements forward and backward definitions of an autograd operation.
#       Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

def method_one():
    print()
    print("Loss Function:")

    # A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.
    # several different loss functions in nn package. MSELoss is an example
    output = net(input)
    target = torch.randn(10)        # a dummy target
    target = target.view(1, -1)     # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)
    print(loss.grad_fn)                         # MSELoss
    print(loss.grad_fn.next_functions[0][0])     # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

    print()
    print(" Backprop")
    net.zero_grad()     # zeroes the gradient buffers of all params
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    print()
    print("Updating the weights")
    # with SGD: weight = weight - learning_rate * gradient
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)


def method_two():
    # using torch.optim I can use different update rules such as Adam, SGD, etc
    import torch.optim as optim
    print()
    print("Own optimizer:")

    target = torch.randn(10)  # a dummy target
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    #create own optimizer
    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    # in your training loop
    optimizer.zero_grad()       # zero gradient buffers
    output = net(input)
    loss = criterion(output, target)

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)
    loss.backward()
    optimizer.step()            # does the update
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

method_one()
method_two()