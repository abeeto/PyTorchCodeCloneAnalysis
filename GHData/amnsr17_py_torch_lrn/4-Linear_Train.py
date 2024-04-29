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


import torch.optim as optim
my_net = MyNet()
optimizer = optim.Adam(my_net.parameters(), lr=0.001)

EPOCHS = 3
for epoch in range(EPOCHS):
    # this data is a batch of examples and their respective labels
    for data in trainset:
        # Examples=X Labels=y
        X, y = data
        # --- Gradient Calculation ---
        # Each time. before passing data through the network, gradients are set to zero for new exampke to pass through the network
        my_net.zero_grad()
        # passing data now, giving -1 for any size but actually we have batch_size = 10
        output = my_net.forward(X.view(-1,28*28))
        # Loss calculation
        loss = F.nll_loss(output, y)
        # back propagation of error
        loss.backward()
        # adjust the weights according to the calculated error
        optimizer.step()
    print(epoch, loss)


# Checking corrected predictions made during the training
correct = 0
total = 0
# we do not want to optimize on this test data . so do not calculate gradients: no_grad()
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = my_net.forward(X.view(-1,28*28))
        for idx, i in enumerate(output):
            # output values
            if torch.argmax(i) == y[idx]:
                correct += 1
            total+=1
print("Accuracy: ", round(correct/total, 3))

# Taking one example and predicting it
import matplotlib.pyplot as plt
plt.imshow(X[1].view(28,28))
plt.show()
# argmax is also returning a list and we want to read its first element
print(torch.argmax(my_net(X[1].view(-1,28*28)[0])))



