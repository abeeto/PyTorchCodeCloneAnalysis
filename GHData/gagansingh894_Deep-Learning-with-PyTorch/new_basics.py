import torch
import numpy as np

def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    :param x:
    :return:
    """
    return 1/(1+torch.exp(-x))


# Generate some data
torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))

# res = activation(torch.mm(features, weights.T) + bias)

# Generate some data
features = torch.randn((1, 3))

n_input = features.shape[1]
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

res = activation(torch.mm(features, W1) + B1)
res = activation(torch.mm(res, W2) + B2)
print(res)


a = np.random.rand(4, 3)
b = torch.from_numpy(a)

# Neural Network using torch.nn

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

train_set = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()
print(type(images))
print(images.shape)
print(labels.shape)

inputs = images.view(images.shape[0], -1)
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)
out = torch.mm(h, w2) + b2
print(out.shape)


def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


probabilities = softmax(out)
print(probabilities.shape)

# Neural Network Architectire
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256,10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x


model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))


images = images.view(images.shape[0], -1)
criterion = nn.NLLLoss()

# Forward Pass
logits = model(images)

# Calculate loss with logits
loss = criterion(logits, labels)
# Autograd -> gradient descent
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

# Optimizer
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
images, labels = next(iter(train_loader))
images = images.resize_(64, 784)
optimizer.zero_grad()
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)
optimizer.step()
print('Gradient -', model[0].weight)


# FI
