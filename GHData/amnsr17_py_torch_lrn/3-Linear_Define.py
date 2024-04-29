import torch
from torchvision import transforms,datasets

# transform -> convert the data to Tensor
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        # instantiating nn.Module
        super().__init__()
        # images are 28*28 = 784 on flattening we need 789 inputs
        # we are making 3 hidden layers of 64 neurons each
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        """ Defines how the data will be moving through the layers i.e. Forward Pass
        :param x: the input data
        :return: output of the data after passing through forward pass
        """
        # output x after passing through first layer fc1
        # applying activation function on the entire layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # this layer is the output and neurons are not firing so we we don't have an activation function here
        x = self.fc4(x)
        # In order to see a probability distribution for multiclass we use log_softmax to have readable distribution
        # dim=1 - makes probability distribution accross the classes as define in the output layer size. i.e. 10 for 10 classes
        return F.log_softmax(x, dim=1)


X = torch.rand(28,28)
# ---- flattening
# -1 is the requirement of the library for saying the batch size can be any. Fo batch size 2 give 2.
X = X.view(-1, 28*28)
output = MyNet().forward(X)
print(output)

#
# my_net = MyNet()
# print(my_net)





