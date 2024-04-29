from typing import Any

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

num_epochs = 5
batch_size = 100
learning_rate = 0.001

train = datasets.MNIST(root='./data',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)

test = datasets.MNIST(root='./data',
                      train=False,
                      transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dense = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


cnn = CNN()

if os.path.isfile('cnn.pkl'):
    cnn.load_state_dict(torch.load('cnn.pkl'))
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            optimizer.zero_grad()
            out = cnn(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d], Loss: %.4f"
                      % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.item()))

    if not os.path.isfile('cnn.pkl'):
        torch.save(cnn.state_dict(), 'cnn.pkl')

cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    out = cnn(images)
    _, pred = torch.max(out.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum()

print("Test Accuracy = %f%%" % (100 * correct / total))
