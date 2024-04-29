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

import pickle
from options import Options, attrs


DEF_BATCH_SIZE = 10
# LEARNING_RATE = 1e-3
# MOMENTUM = 0.5
LOG_INTERVAL = 100
# # LOSS_FUNCTION = {'nll': 1, 'mse': 2}
# LOSS_FUNCTION = 'mse'  # 'mse'


# Global paramerets...
class Global_Params(object):

    options = Options(
        epochs        = 1,
        batch_size    = DEF_BATCH_SIZE,
        learning_rate = 1e-3,
        momentum      = 0.5,
        log_interval  = 100,
        loss_function = 'nll',
    )

    def __init__(self, **kwargs):
        self.options = Global_Params.options.push(kwargs)

    def set_opts(self, **kwargs):
        self.options.set(**kwargs)

    def self_print(self):
        print(attrs(self.options))


class Model(nn.Module):
    options = Options(
        loss_function = 'nll'
    )

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)
        self.options = Model.options.push(kwargs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if (self.options.loss_function == 'nll'):
            return F.log_softmax(x, dim=1)
        if (self.options.loss_function == 'mse'):
            return F.softmax(x, dim=1)
        return x

    def set_opts(self, **kwargs):
        self.options.set(**kwargs)

    def self_print(self):
        print(self.options.loss_function)


def convert_y_batch(y_batch, net):
    if (net.options.loss_function == 'mse'):
        y_batch_temp = torch.zeros(DEF_BATCH_SIZE, 10)
        for y_idx, y in enumerate(y_batch):
            y_batch_temp[y_idx][y] = 1.
        return y_batch_temp
    if (net.options.loss_function == 'nll'):
        return y_batch


def train(args, net, device, train_data_set, optimizer, criterion, epoch, train_losses, train_counter):
    net.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_data_set):
        # Prepare 'x' and 'y' batchs...
        x_batch = x_batch.view(-1, 28 * 28).to(device)
        y_batch = convert_y_batch(y_batch, net).to(device)

        # Net model running...
        y_pred = net(x_batch)

        # Loss calculation and training...
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Debug and statistic print...
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


def test(args, net, device, test_data_set, criterion, test_losses):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_data_set:
            # Prepare 'x' and 'y' batchs...
            x_batch = x_batch.view(-1, 28 * 28).to(device)
            y_batch = convert_y_batch(y_batch, net).to(device)

            # Net model running...
            y_pred = net(x_batch)

            # Loss calculation for log only...
            test_loss += criterion(y_pred, y_batch, size_average=False).item()
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(y_batch.data.view_as(pred)).sum().item()

            # test_loss += criterion(y_pred, y_batch, reduction='sum').item()  # sum up batch loss
            # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(y_batch.view_as(pred)).sum().item()

    # Debug and statistic print...
    test_loss /= len(test_data_set.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_data_set.dataset),
        100. * correct / len(test_data_set.dataset))
    )


def main():
    # Input parameters parsing and setting...
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=1,     metavar='N',         help='number of epochs to train (default: 1)')
    parser.add_argument('--loss',   type=str, default='nll', metavar='N',         help='criterion function: \'nll\' or \'mse\'  (default: nll)')
    parser.add_argument('--save',             default=False, action='store_true', help='save training results (default: false)')
    parser.add_argument('--load',             default=False, action='store_true', help='load training results and skeep training (default: false)')
    args = parser.parse_args()

    if (args.loss != 'mse') and (args.loss != 'nll'):
        print("Wrong loss function! NLL will be used!")
        args.loss = 'nll'
    global_params = Global_Params()
    global_params.self_print()
    params = global_params.options

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Dataset loading...
    train_loader = datasets.MNIST('.', train=True,  transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_loader  = datasets.MNIST('.', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

    train_data_set = DataLoader(train_loader, batch_size=params.batch_size, shuffle=True,  **kwargs)
    test_data_set  = DataLoader(test_loader,  batch_size=params.batch_size, shuffle=False, **kwargs)

    # Neural network instantiation...
    net = Model(loss_function=args.loss).to(device)
    print(net)
    net.self_print()

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    if (args.loss == 'mse'):
        criterion = F.mse_loss
    if (args.loss == 'nll'):
        criterion = F.nll_loss

    train_counter = []
    train_losses  = []
    test_counter  = [(i + 1) * len(train_data_set.dataset) for i in range(args.epochs)]
    test_losses   = []

    # Tarining and testing...
    if (args.load):
        net.load_state_dict(torch.load('./results/model.pth'))
        test(0, net, device, test_data_set, criterion, test_losses)
    else:
        for epoch in range(args.epochs):
            train(0, net, device, train_data_set, optimizer, criterion, epoch, train_losses, train_counter)
            if (args.save):
                # Save model and optimizer every training epoch...
                file_name = './results/model_%s.%02d.pth' % (args.loss, epoch)
                torch.save(net.state_dict(),       file_name)
                file_name = './results/optimizer_%s.%02d.pth' % (args.loss, epoch)
                torch.save(optimizer.state_dict(), file_name)
                # Save (rewrite) full training model for recovery...
                torch.save(net.state_dict(),       './results/model_%s.pth' % args.loss)
            # test(0, net, device, test_data_set, criterion, test_losses)

    # Log output...

    # # print(train_counter)
    # # print(train_losses)
    # # print(test_counter)
    # # print(test_losses)

    with open('./logs/train_counter_%s' % args.loss, 'wb') as fp:
        pickle.dump(train_counter, fp)
    with open('./logs/train_losses_%s' % args.loss, 'wb') as fp:
        pickle.dump(train_losses, fp)
    with open('./logs/test_counter_%s' % args.loss, 'wb') as fp:
        pickle.dump(test_counter, fp)
    with open('./logs/test_losses_%s' % args.loss, 'wb') as fp:
        pickle.dump(test_losses, fp)

    # fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    # plt.plot(test_counter,  test_losses,  color='red')
    # plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == '__main__':
    main()
