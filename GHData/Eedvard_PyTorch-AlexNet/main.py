import os, time
import torch
import requests, zipfile, sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils, datasets
import random, matplotlib
import pandas as pd
from torchvision.models.resnet import BasicBlock
from torch.utils.tensorboard import SummaryWriter

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)

        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)

        x = self.relu(x)
        x = self.conv4(x)

        x = self.relu(x)
        x = self.conv5(x)

        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

def GetDataLoaders():
    # Initializing the transformations for the data
    # For the test data the horizontal flip is not required.

    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    test_transform = transforms.Compose([transforms.Resize(227),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])
    dataset = datasets.FashionMNIST('FMNIST_data/', download=True, train=True, transform=transform)

    # Split the data for training set and validation set
    trainset, valset = torch.utils.data.random_split(dataset, [50000, 10000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True)

    # Load the testing data
    testset = datasets.FashionMNIST('FMNIST_data/', download=True, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    return trainloader, validloader, testloader


def eval(net, loss_function, data_loader):
    net.eval()
    correct = 0.0
    num_images = 0.0
    running_loss = 0.0
    for i, sample in enumerate(data_loader):
        images, labels = sample
        outs = net(images)
        _, preds = outs.max(1)
        correct += preds.eq(labels).sum()
        running_loss += loss_function(outs, labels).item()
        num_images += len(labels)

    acc = correct.float() / num_images
    loss = running_loss / len(data_loader)
    return acc, loss


def train(net, train_loader, valid_loader, writer, val_writer, loss_function):
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    log_every = 100
    epoches = 10
    last_val = 0
    for epoch in range(epoches):
        start_t = time.time()
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, sample in enumerate(train_loader):
            images, labels = sample
            outs = net(images)
            loss = loss_function(outs, labels)
            _, preds = outs.max(1)
            correct = preds.eq(labels).sum()

            running_acc += correct.float() / len(labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % log_every == 99:
                print('[Epoch/iter]: [{}/{}], loss: {:.05f}, accuracy: {:.05f}'.format(epoch, i + 1,
                                                                                       running_loss / log_every,
                                                                                       running_acc / log_every))

                log_index = epoch * len(train_loader) + i
                writer.add_scalar('Loss', running_loss / log_every, log_index)
                writer.add_scalar('Accuracy', running_acc / log_every, log_index)
                running_loss = 0.0
                running_acc = 0.0

        acc_eval, loss_eval = eval(net, loss_function, valid_loader)
        print('Elapsed time: {:.02f} seconds, end of epoch: {}, lr: {}, val_loss: {:.05f}, val_acc: {:.05f}'.format(
            time.time() - start_t, epoch, optimizer.param_groups[0]['lr'], loss_eval, acc_eval))
        val_writer.add_scalar('Loss', loss_eval, log_index)
        val_writer.add_scalar('Accuracy', acc_eval, log_index)
        if (
                loss_eval > last_val and last_val != 0):  # If the validation loss stops improving the learning rate is divided by 10.
            last_val = loss_eval
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10

    return net

def test(net, testloader, loss_function):

    acc_test, loss_test = eval(net, loss_function, testloader)
    print('Accuracy on testing data: {:.05f}'.format(acc_test))

def main():
    # Initialize tensorboard writers to visualize the training process
    writer = SummaryWriter('graphs/training')
    val_writer = SummaryWriter('graphs/validation')

    PATH = './net.pth'
    net = Network(num_classes=10)

    loadnet = True  # Change this to false if don't want to load the existing trained net

    if (loadnet):
        if os.path.exists(PATH):
            net.load_state_dict(torch.load(PATH))
            net.eval()
            print("Checkpoint loaded")
        else:
            print("No checkpoint found")

    trainloader, validloader, testloader = GetDataLoaders()
    print('Beginning to train the network')
    loss_function = torch.nn.CrossEntropyLoss()
    trained_net = train(net, trainloader, validloader, writer, val_writer, loss_function)

    PATH = './net.pth'
    torch.save(net.state_dict(), PATH)

    writer.close()
    val_writer.close()

    test(net, testloader, loss_function)
if __name__ == "__main__":
    main()