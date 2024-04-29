import torch
import random
import numpy as np

# fix seed for using the same initial weights
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# get datasets
import torchvision.datasets

numbers_train = torchvision.datasets.MNIST('./', download=True, train=True)
numbers_test = torchvision.datasets.MNIST('./', download=True, train=False)

# separate into features and predicted value

x_train = numbers_train.train_data
y_train = numbers_train.train_labels

x_test = numbers_test.test_data
y_test = numbers_test.test_labels

# make train set useful
x_train = x_train.float()  # [60000, 28, 28]
x_test = x_test.float()

"""
import matplotlib.pyplot as plt
plt.imshow(x_train[0, :, :])
plt.show()
print(y_train[0])
"""

# make artificial dimension for Conv2D

x_test = x_test.unsqueeze(1).float()
x_train = x_train.unsqueeze(1).float()


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # make padding  = 2 for getting output as 28x28
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = torch.nn.Tanh()

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.Tanh()

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


net = LeNet5()

# use cross-entropy like loss - function
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
