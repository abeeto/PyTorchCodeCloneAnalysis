import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as utils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings

from Constants.constants import EMOTIONS, DATA_PATH


warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def label_classes():
    """
    Dictionary for labels
    """
    return tuple(EMOTIONS)


def load_data():
    """
    Load dataset from numpy(.npy) objects
    : Reshape to make numpy array a pytorch tensor
    """
    # pytorch images are represented as [batch_size, channels, height, width]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=transform
    )
    train_dataloader = utils.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    test_dataloader = utils.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    return train_dataloader, test_dataloader


def imshow(img):
    """
    Function to print pytorch tensor image
    : npy images are normalized by a factor of 255
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    """
    Architecture of CNN
    Input Image Size: 48*48
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(5, 5)
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 5)
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(4, 4)
        )
        self.fc1 = nn.Linear(128*5*5, 3072)
        self.fc2 = nn.Linear(3072, len(EMOTIONS))

    def forward(self, x):
        """
        Connections within the Architecture
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


# Globally defining the Architecture
net = Net()


def Optimizer():
    """
    Optimization Criteria
    : Categorical Crossentropy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


def train():
    """
    Script to train model
    """
    trainloader, _ = load_data()
    classes = label_classes()
    # Random check training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Actual model training
    print('Start Training')
    criterion, optimizer = Optimizer()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def test():
    """
    Script to test model
    """
    _, testloader = load_data()

    # Random check training images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

    # Actual model testing
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def check_prediction():
    """
    Function to check prediction for test images
    """
    _, inputs, _, labels = load_data()
    images = inputs[:4]
    imshow(torchvision.utils.make_grid(images))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)


def main():
    """
    Main Function from which script is executed
    """
    train()


main()
