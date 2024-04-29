"""
Run similar model as in lab 4

modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from util import plot_confusion_matrix

torch.manual_seed(0)

def cifar100(seed):
    np.random.seed(seed)
    selected_cats = np.random.choice(np.arange(0, 100), 5, replace=False)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                             download=True, transform=transform)

    trainset.targets = np.array(trainset.targets)
    selected_train = np.isin(trainset.targets, selected_cats)
    trainset.targets = list(trainset.targets[selected_train])
    _, trainset.targets = np.unique(trainset.targets,
                                    return_inverse=True)
    trainset.data = trainset.data[selected_train]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False,
                                            download=True, transform=transform)

    testset.targets = np.array(testset.targets)
    selected_test = np.isin(testset.targets, selected_cats)
    testset.targets = list(testset.targets[selected_test])
    _, testset.targets = np.unique(testset.targets,
                                   return_inverse=True)
    testset.data = testset.data[selected_test]

    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, x):

        raise StandardError

    def fit(self, trainloader):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # setup SGD
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.0)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute forward pass
                outputs = self.forward(inputs)

                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics3
                running_loss += loss.item()

            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i+1)))

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        with torch.no_grad():
            for images, labels in testloader:
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predicted += predicted.numpy().tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        return all_predicted


class ConvNet(BaseNet):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(1152, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1152)
        x = self.fc1(x)
        return x


class FullNet(BaseNet):
    def __init__(self):
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x



if __name__ == '__main__':
    # get data
    trainloader, testloader = cifar100(1337)

    # full net
    print ("Fully connected network")
    net1 = FullNet()
    net1.fit(trainloader)
    pred_labels = net1.predict(testloader)
    plt.figure(1)
    test_labels = testloader.dataset.targets
    plot_confusion_matrix(pred_labels, test_labels, "FullNet")
    # plt.show()

    # conv net
    print ("Convolutional network")
    net2 = ConvNet()
    net2.fit(trainloader)
    pred_labels = net2.predict(testloader)
    plt.figure(2)
    plot_confusion_matrix(pred_labels, test_labels, "ConvNet")

    plt.show()
