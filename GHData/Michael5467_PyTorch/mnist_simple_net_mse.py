#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np

DEF_BATCH_SIZE = 10
LEARNING_RATE = 1e-3
MOMENTUM = 0.5
LOG_INTERVAL = 100


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def train(args, net, device, train_data_set, optimizer, criterion, epoch, train_losses, train_counter, one_hot):
    net.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_data_set):
        # print("y_batch")
        # print(y_batch)

        y_batch_tensor = torch.empty(DEF_BATCH_SIZE, 10)
        # print("y_batch_tensor = \n", y_batch_tensor)
        for y_idx, y in enumerate(y_batch):
            y_batch_tensor[y_idx] = one_hot[y]

        # print("new y_batch_tensor")
        # print(y_batch_tensor.size())
        # print(y_batch_tensor)

        x_batch = x_batch.view(-1, 28 * 28).to(device)
        # y_batch = y_batch.to(device)
        y_batch_tensor = y_batch_tensor.to(device)
        y_pred = net(x_batch)

        # print(y_pred.size())
        # print(y_pred)

        loss = criterion(y_pred, y_batch_tensor)
        # loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_batch),
                len(train_data_set.dataset),
                100. * batch_idx / len(train_data_set),
                loss.item())
            )
            train_counter.append((batch_idx * DEF_BATCH_SIZE) + ((epoch) * len(train_data_set.dataset)))
            train_losses.append(loss.item())
            # torch.save(net.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')


def test(args, net, device, test_data_set, criterion, test_losses, one_hot):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_data_set:

            x_batch = x_batch.view(-1, 28 * 28).to(device)
            y_batch_tensor = torch.empty(DEF_BATCH_SIZE, 10)
            for y_idx, y in enumerate(y_batch):
                y_batch_tensor[y_idx] = one_hot[y]
            y_batch = y_batch.to(device)
            y_batch_tensor = y_batch_tensor.to(device)
            y_pred = net(x_batch)

            test_loss += criterion(y_pred, y_batch_tensor).item()
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(y_batch.data.view_as(pred)).sum()
    test_loss /= len(test_data_set.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_data_set.dataset),
        100. * correct / len(test_data_set.dataset))
    )


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 1)')
    args = parser.parse_args()

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = datasets.MNIST('.', train=True,  transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_loader  = datasets.MNIST('.', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

    train_data_set = DataLoader(train_loader, batch_size=DEF_BATCH_SIZE, shuffle=True,  **kwargs)
    test_data_set  = DataLoader(test_loader,  batch_size=DEF_BATCH_SIZE, shuffle=False, **kwargs)

    net = Model().to(device)
    print(net)
    one_hot = torch.eye(10)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # criterion = F.nll_loss
    criterion = F.mse_loss

    train_counter = []
    train_losses  = []
    test_counter  = [(i + 1) * len(train_data_set.dataset) for i in range(args.epochs)]
    test_losses   = []

    # train(0, net, device, train_data_set, optimizer, criterion, 0, train_losses, train_counter)
    # test(0, net, device, test_data_set, criterion, test_losses, one_hot)
    for epoch in range(args.epochs):
        train(0, net, device, train_data_set, optimizer, criterion, epoch, train_losses, train_counter, one_hot)
        test(0, net, device, test_data_set, criterion, test_losses, one_hot)

    # print(train_counter)
    # print(train_losses)
    # print(test_counter)
    # print(test_losses)

    # fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter,  test_losses,  color='red')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == '__main__':
    main()
