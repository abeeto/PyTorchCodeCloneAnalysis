import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# number of subprocesses to use for data loading
num_workers = 0  # means to use all
# how many samples per batch to load
batch_size = 64
# where the dataset is:
dataset_path = "./MNIST"

# convert data to torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# create training and test datasets
train_data = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)

# create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.linear1 = torch.nn.Linear(20 * 4 * 4, 50)
        self.predict = torch.nn.Linear(50, 10)

    def forward(self, x):
        # Define the forward pass operations here
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(F.dropout(self.linear1(x)))
        x = self.predict(x)
        return x


model = Net()

# specify loss function
criterion = nn.CrossEntropyLoss()  # modify that

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # modify that

# number of epochs to train the model
n_epochs = 50
train_loss_progress = []
test_accuracy_progress = []

model.train()  # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for batch_idx, (data, target) in enumerate(train_loader):
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(data)
        # calculate the loss
        loss = criterion(outputs, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # if you have a learning rate scheduler - perform a its step in here

        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    train_loss_progress.append(train_loss)
    correct = 0
    total = 0
    model.eval()  # prep model for testing

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    test_loss = (100 * correct / total)
    test_accuracy_progress.append(test_loss)

# Plotting:
x_range = np.arange(1, n_epochs + 1)
fig, axs = plt.subplots(2)
axs[0].plot(x_range, train_loss_progress, c='b', label="Train loss")
axs[1].plot(x_range, test_accuracy_progress, c='r', label="Test accuracy")
axs[0].legend()
axs[1].legend()
plt.show()