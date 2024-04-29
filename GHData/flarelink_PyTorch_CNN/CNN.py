# =============================================================================
# CNN.py - CNN implementation using PyTorch testing randomness
# Copyright (C) 2018  Humza Syed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
from distutils.dir_util import copy_tree
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import random
plt.ion()


# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device, " the GPU device is selected")
else:
    device = torch.device('cpu')
    print(device, " the CPU device is selected")
    


def load_data_MNIST(batch_size):
    """
    Load MNIST dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for MNIST
    """

    transform = transforms.Compose([
        transforms.Resize(size=(28, 28)),
        transforms.ToTensor()
        ])

    trainset = torchvision.datasets.MNIST(root='./MNIST', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./MNIST', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def load_data_CIFAR10(batch_size):
    """
    Load CIFAR-10 dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for CIFAR-10
    """

    transform = transforms.Compose([
        transforms.Resize(size=(28, 28)),
        transforms.ToTensor()
        ])

    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


"""
##############################################################################
# Functions to define weight initialization
##############################################################################
"""
def weight_switcher(weight_type, nf, filter_d, in_chan):
    """
    Determines what weight type to return
    :param weight_type: selected weight type distribution
    :param nf:          number of filters
    :param filter_d:    filter size
    :param in_chan:     number of color channels
    :return: weights as a torch for Pytorch
    """

    def normal(nf, filter_d, in_chan):
        """
        Normalized weight distribution with mean of 0 and standard deviation of 0.05

        :param nf:          number of filters
        :param filter_d:    filter size
        :param in_chan:     number of color channels
        :return: weight distribution
        """
        return np.random.normal(loc=0.0, scale=0.05, size=(nf, in_chan, filter_d, filter_d))

    def laplace(nf, filter_d, in_chan):
        """
        Laplace weight distribution with mean of 0 and decay of 0.05

        :param nf:          number of filters
        :param filter_d:    filter size
        :param in_chan:     number of color channels
        :return: weight distribution
        """
        return np.random.laplace(loc=0.0, scale=0.05, size=(nf, in_chan, filter_d, filter_d))

    def logistic(nf, filter_d, in_chan):
        """
        Logistic weight distribution

        :param nf:          number of filters
        :param filter_d:    filter size
        :param in_chan:     number of color channels
        :return: weight distribution
        """
        return np.random.logistic(loc=0.0, scale=0.05, size=(nf, in_chan, filter_d, filter_d))

    def uniform(nf, filter_d, in_chan):
        """
        Uniform weight distribution

        :param nf:          number of filters
        :param filter_d:    filter size
        :param in_chan:     number of color channels
        :return: weight distribution
        """
        return np.random.uniform(low=-0.05, high=0.05, size=(nf, in_chan, filter_d, filter_d))


    def Gaussian_with_correlation(nf, filter_d, in_chan, r):
        """
        Gaussian correlaton weight distribution

        :param nf:          number of filters
        :param filter_d:    filter size
        :param in_chan:     number of color channels
        :param r:           determines correlation for Gaussian; if desired
                            correlation = 0, then r = 0.0, if desired
                            correlation = 1, then r = 0.1, etc. until 9
                            exception is only for correlation = 10, then r =
                            0.99
        :return: weight distribution
        """

        assert (r < 1.0)

        Filters = np.zeros((nf, filter_d, filter_d, in_chan), dtype=float)
        Filters_mean = np.zeros((nf), dtype=float)
        Filters_std = np.zeros((nf), dtype=float)
        Filters_std2 = np.zeros((nf), dtype=float)
        mean_matrix = np.zeros((nf))
        Cov_matrix = np.zeros((nf, nf))
        Wnew = np.zeros((filter_d, filter_d, in_chan, 64))

        for i in range(nf):
            Filters[i] = np.random.normal(loc=0.0, scale=0.05, size=(filter_d, filter_d, in_chan))
            Filters_mean[i] = np.mean(Filters[i])
            Filters_std[i] = np.std(Filters[i])
            Filters_std2[i] = Filters_std[i] ** 2

        mean_matrix = Filters_mean
        for i in range(nf):
            for j in range(nf):
                if (i == j):
                    Cov_matrix[i][j] = float(Filters_std2[i])
                elif (j > i):
                    Cov_matrix[i][j] = float(r * (Filters_std[i] * Filters_std[j]))
                else:
                    Cov_matrix[i][j] = float(-r * (Filters_std[i] * Filters_std[j]))

        # print(Cov_matrix)
        Wnew = np.random.multivariate_normal(mean_matrix, Cov_matrix, (filter_d, filter_d, in_chan))
        print(Wnew.shape)
        print('weight shape')
        Wnew = Wnew.flatten()
        Wnew = np.reshape(Wnew, (nf, in_chan, filter_d, filter_d))
        print(Wnew.shape)
        print('new weight shape')
        # print(Wnew)

        return Wnew

    switcher = {
        'norm':     lambda: normal(nf, filter_d, in_chan),
        'lap':      lambda: laplace(nf, filter_d, in_chan),
        'log':      lambda: logistic(nf, filter_d, in_chan),
        'unif':     lambda: uniform(nf, filter_d, in_chan),
        'Cory':     lambda: Cory_style_Sin_gen1(nf, filter_d, in_chan),
        'Gaus_0':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.0),
        'Gaus_1':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.1),
        'Gaus_2':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.2),
        'Gaus_3':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.3),
        'Gaus_4':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.4),
        'Gaus_5':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.5),
        'Gaus_6':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.6),
        'Gaus_7':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.7),
        'Gaus_8':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.8),
        'Gaus_9':   lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.9),
        'Gaus_10':  lambda: Gaussian_with_correlation(nf, filter_d, in_chan, r=0.99)
    }

    weights = switcher.get(weight_type, lambda: 'Invalid data, out of bounds!')
    weights = torch.from_numpy(weights()).float()

    return weights


def outputSize(in_size, kernel_size, stride, padding):
    """
    Help function to calculate output size from conv layer
    :param in_size:     image size, ex: 28 x 28 image means in_size = 28
    :param kernel_size: filter size
    :param stride:      stride for conv operator
    :param padding:     padding for conv
    :return: output size for conv layer
    """
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return output


"""
##############################################################################
# CNN classes for network with 1 conv layer and a newtork with 2 conv layers
##############################################################################
"""
class CNN_1Lay(nn.Module):
    """
    CNN with 1 conv layer and 1 fully connected layer, conv layer's weights are frozen
    """
    # Convolutional neural network (one convolutional layers)
    def __init__(self, nf=64, filter_d=3, weights=None, in_chan=1, num_classes=10):
        super(CNN_1Lay, self).__init__()

        # CNN creation
        if (weights is None):
            print('weight type not specified')

        self.conv1 = nn.Conv2d(in_channels=in_chan,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv1.weight = nn.Parameter(weights)
        self.conv1.weight.requires_grad = False
        self.fc = nn.Linear(28 * 28 * nf, num_classes)  # 10 classes

        # print('The output size would be: {:d}'.format(outputSize(28, 3, 1, 1)))

    # CNN forward pass
    def forward(self, x):
        # print('x shape before conv')
        # print(x.size())
        x = F.relu(self.conv1(x))
        # print('x shape after conv')
        # print(x.shape)
        x = x.view(-1, 28 * 28 * 64)
        x = self.fc(x)
        return x


class CNN_2Lay(nn.Module):
    """
    CNN with 2 conv layers and 1 fully connected layer, conv layers' weights are frozen
    """
    # Convolutional neural network (two convolutional layers)
    def __init__(self, nf=64, filter_d=3, weights=None, in_chan=1, num_classes=10):
        super(CNN_2Lay, self).__init__()

        if (weights is None):
            print('weight type not specified')

        self.conv1 = nn.Conv2d( in_channels=in_chan,    # 1 input channel since b/w
                                out_channels=nf,  # number of filters
                                kernel_size=filter_d,    # filter size
                                stride=1,         # stride of 1
                                padding=1)        # padding of 1
        self.conv1.weight = nn.Parameter(weights)
        self.conv1.weight.requires_grad = False

        self.conv2 = nn.Conv2d(in_channels=nf,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv2.weight.requires_grad = False
        # self.conv1.weight = nn.Parameter(weights)

        self.fc = nn.Linear(28 * 28 * nf, num_classes) # 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        x = x.view(-1, 28 * 28 * 64)
        x = self.fc(x)
        return x


class CNN_1Lay_Split(nn.Module):
    """
    CNN with 2 conv layers merged to form 1 conv layer where 50% trained and 50% random
    and 1 fully connected layer
    """
    # Convolutional neural network (two convolutional layers)
    def __init__(self, nf=32, filter_d=3, weights=None, in_chan=1, num_classes=10):
        super(CNN_1Lay_Split, self).__init__()

        # CNN creation
        if (weights is None):
            print('weight type not specified')

        self.conv1 = nn.Conv2d(in_channels=in_chan,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv1.weight = nn.Parameter(weights)
        self.conv1.weight.requires_grad = False

        self.conv2 = nn.Conv2d(in_channels=in_chan,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv2.weight = nn.Parameter(weights)
        self.conv2.weight.requires_grad = True  

        self.fc = nn.Linear(28 * 28 * nf * 2, num_classes)  # 10 classes

        # print('The output size would be: {:d}'.format(outputSize(28, 3, 1, 1)))

    # CNN forward pass
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = torch.cat((c1, c2), 0)

        x = F.relu(c3)
        # print(x.shape)
        x = x.view(-1, 28 * 28 * 64)
        x = self.fc(x)
        return x


class CNN_1Lay_FullTrain(nn.Module):
    """
    CNN with 1 conv layer fully trained and 1 fully connected layer
    """
    # Convolutional neural network (two convolutional layers)
    def __init__(self, nf=64, filter_d=3, weights=None, in_chan=1, num_classes=10):
        super(CNN_1Lay_FullTrain, self).__init__()

        # CNN creation
        if (weights is None):
            print('weight type not specified')

        self.conv1 = nn.Conv2d(in_channels=in_chan,  # 1 input channel since b/w
                               out_channels=nf,  # number of filters
                               kernel_size=filter_d,  # filter size
                               stride=1,  # stride of 1
                               padding=1)  # padding of 1
        self.conv1.weight = nn.Parameter(weights)
        self.conv1.weight.requires_grad = True

        self.fc = nn.Linear(28 * 28 * nf, num_classes)  # 10 classes

        # print('The output size would be: {:d}'.format(outputSize(28, 3, 1, 1)))

    # CNN forward pass
    def forward(self, x):
        c1 = self.conv1(x)
        x = F.relu(c1)
        # print(x.shape)
        x = x.view(-1, 28 * 28 * 64)
        x = self.fc(x)
        return x


"""
##############################################################################
# Training and testing
##############################################################################
"""
def training(num_epochs, batch_size, train_data, model, learning_rate, decay, log_file, min_delta, patience):
    """
    Training for random CNN

    :param num_epochs:    number of epochs to train for
    :param batch_size:    batch size defined for incoming data
    :param train_data:    the train data input images
    :param model:         the CNN model
    :param learning_rate: learning rate for ADAM optimizer
    :param decay:         decay for ADAM optimizer
    :param log_file:      logging file to be printed to
    :param min_delta:     parameter for early stopping (WIP)
    :param patience:      parameter for early stopping (WIP)
    :return: train accuracy, train time, list of loss values over epochs,
             and list of training & testing accuracies over epochs
    """
    total = 0
    correct = 0
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    temp_loss = 0.0
    logging_time = 0.0
    num_bad_epochs = 0

    # loss and optimizer
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=learning_rate, weight_decay=decay)

    # number of batches
    n_batches = len(train_data)
    print('The batch size is {:d}'.format(batch_size))


    start_to_train = time.time()

    for epoch in range(num_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        total_train_loss = 0
        train_start_loop = time.time()

        for i, data in enumerate(train_data, 0):

            # get inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # setting gradient to zero
            optimizer.zero_grad()

            # forward pass and optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # capture end of loop time
            training_end_loop = time.time()

            # print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            # print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, training_end_loop - train_start_loop))
                log_file.write("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s \n".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, training_end_loop - train_start_loop))
                temp_loss = running_loss
                # Reset running loss and time
                running_loss = 0.0
                train_start_loop = time.time()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        temp_time = time.time()

        print('Train Accuracy of the model at epoch {:d} on the {:d} train images: {} %'.format(epoch, n_batches * batch_size, 100 * correct / total))
        log_file.write('Train Accuracy of the model at epoch {:d} on the {:d} train images: {} % \n'.format(epoch, n_batches * batch_size, 100 * correct / total))

        # obtaining loss and training accuracies over time
        loss_list.append(temp_loss)
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        # obtaining test accuracies over time
        test_acc, test_time = testing(test_data, model, log_file)
        test_acc_list.append(test_acc)

        best_test_acc = max(test_acc_list)
        stop, num_bad_epochs = early_stopping(min_delta, patience, test_acc, best_test_acc, num_bad_epochs)

        if(stop):
            print('Early stopping initiated')
            break

        test_temp_time = time.time()

        # capture logging time and subtract from training time
        logging_time += test_temp_time - temp_time

    ender = time.time()
    end_of_train = (ender - start_to_train - logging_time) / 60

    print('Total training time was: {:.2f}mins'.format(end_of_train))
    log_file.write('Total training time was: {:.2f}mins \n'.format(end_of_train))

    train_acc_final = 100 * correct / total

    return train_acc_final, end_of_train, loss_list, train_acc_list, test_acc_list


def testing(test_data, model, log_file):
    """
    Testing for random CNN

    :param test_data: test data input images
    :param model:     the CNN model
    :param log_file:  logging file to be printed to
    :return: test accuracy and test time
    """

    start_to_test = time.time()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    test_len = len(test_data)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_data, 0):

            # get inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {:d} test images: {} %'.format(test_len, 100 * correct / total))
        log_file.write('Test Accuracy of the model on the {:d} test images: {} % \n'.format(test_len, 100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
    ender = time.time()
    end_of_test = (ender - start_to_test) / 60

    test_acc = 100 * correct / total

    return test_acc, end_of_test


def early_stopping(min_delta, patience, curr_acc, best_acc, num_bad_epochs):
    """
    Performs early stopping when accuracies aren't changing
    TODO - Needs to be fixed

    :param min_delta:      minimum change between the best accuracy and current accuracy
    :param patience:       how many epochs to check for till stopping
    :param curr_acc:       the current accuracy
    :param best_acc:       the best accuracy
    :param num_bad_epochs: number of bad epochs seen
    :return: stop variable to say if we should stop training and number of bad epochs
    """
    stop = False

    # check that patience is a valid int above 0
    if patience == 0 or patience < 0 or type(patience) != int:
        raise ValueError('Patience must be a valid integer above 0')

    if( ((best_acc - curr_acc) >= min_delta) or (best_acc == curr_acc) ):
        num_bad_epochs = 0
    else:
        num_bad_epochs += 1

    # check that num_bad_epochs is greater than the patience
    if(num_bad_epochs >= patience):
        stop = True

    return stop, num_bad_epochs


def create_parser():
  """
  return: parser inputs
  """
  parser = argparse.ArgumentParser(
        description='PyTorch Random CNN implementation.')

  # arguments for logging
  parser.add_argument('--log_name', type=str, default='testing',
                        help='Specify log name, if None then program will not run; default=None')

  # arguments for dataset and data
  parser.add_argument('--MNIST', type=str2bool, nargs='?', default=True,
                        help='Uses the MNIST dataset if set to True; default=True')
  parser.add_argument('--CIFAR10', type=str2bool, nargs='?', default=False,
                        help='Uses the CIFAR-10 dataset if set to True; default=False')

  # arguments for convolutions in CNN
  parser.add_argument('--nf', type=int, default=64,
                        help='Number of filters for conv layers; default=64')
  parser.add_argument('--filter_d', type=int, default=3,
                        help='Size of filters for conv layers; default=3')
  parser.add_argument('--t_conv', type=int, default=1,
                        help='Chooses conv type 1 - single random conv layer\
                                                2 - two random conv layers \
                                                3 - half train half random single conv layer \
                                                4 - fully trained single conv layer \
                                                default=1')

  # arguments for hyperparameters
  parser.add_argument('--n_epochs', type=int, default=50,
                        help='Defines num epochs for training; default=50')
  parser.add_argument('--n_runs', type=int, default=1,
                        help='Defines num runs for program; default=1')
  parser.add_argument('--batch_size', type=int, default=16,
                        help='Defines batch size for data; default=16')
  parser.add_argument('--lr', type=float, default=0.001,
                        help='Defines learning rate for training; default=0.001')
  parser.add_argument('--decay', type=float, default=0.0004,
                        help='Defines decay for training; default=0.0004')
  parser.add_argument('--delta', type=float, default=0.001,
                        help='Defines min delta for early stopping; default=0.001')
  parser.add_argument('--patience', type=int, default=30,
                        help='Defines patience for early stopping; default=30')
  parser.add_argument('--random_seed', type=int, default=7,
                        help='Defines random seed value; default=7')

  args = parser.parse_args()

  return args


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
if __name__ == '__main__':
    """
    Runs through random CNN and records program time for both training and testing
    """

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    # parsing input arguments
    args = create_parser()

    # Hyper parameters, commented out variables are now in parser
    # batch_size        = 16
    # learning_rate     = 0.001
    # decay             = 0.0004
    # random_seed         = 7
    # num_epochs        = 50
    # num_of_runs       = 1

    # Regular variables, commented out variables are now in parser
    # nf = 64
    # filter_d = 3

    if (args.random_seed == None):
         args.random_seed = random.randint(1, 1000)

    # set reproducible random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # checking dataset for logging
    if(args.MNIST == True):
        in_chan = 1
        dataset = 'MNIST'
    elif(args.CIFAR10 == True):
        in_chan = 3
        dataset = 'CIFAR10'
    # Basic check to see if any dataset was even selected
    else:
        raise IOError('No dataset is selected')

    if(args.log_name == None):
        raise ValueError('A log file name must be specified using --log_name <name_for_log>')

    # open log files and fill in hyperparameters
    today_time = str(datetime.today()).replace(':', '_').replace(' ', '_')
    csv_file = open('./logs/log_run_{}_{}.csv'.format(today_time, args.log_name), 'w+')
    csv_file.write('dataset, conv_layer_type, num_epochs, batch_size, learning_rate, random_seed, decay, num_of_runs, current_run, weight_type, train_acc (%), test_acc (%), train time (mins), test time (mins) \n')
    log_file = open('./logs/log_run_{}_{}.txt'.format(today_time, args.log_name), 'w+')
    log_file.write('dataset_type   = {} \n'.format(dataset))
    log_file.write('conv_layer_type= {:d} \n'.format(args.t_conv))
    log_file.write('num_epochs     = {:d} \n'.format(args.n_epochs))
    log_file.write('batch_size     = {:d} \n'.format(args.batch_size))
    log_file.write('learning_rate  = {} \n'.format(args.lr))
    log_file.write('random_seed    = {:d} \n'.format(args.random_seed))
    log_file.write('decay          = {} \n'.format(args.decay))
    log_file.write('num_of_runs    = {:d} \n \n'.format(args.n_runs))


    # Run through program for a number of times
    for runs in range(args.n_runs):
        log_file.write('Current run is: {:d} \n'.format(runs))

        # start timer
        start_time = time.time()
        # obtain dataloaders of train and test for dataset
        if(args.MNIST == True):
            train_data, test_data = load_data_MNIST(args.batch_size)
        elif(args.CIFAR10 == True):
            train_data, test_data = load_data_CIFAR10(args.batch_size)
        # Basic check to see if any dataset was even selected
        else:
            raise IOError('No dataset is selected')
        log_file.write('dataloaders created successfully \n \n')

        # define weight distribution
        weight_types = [
                        'norm'#,
                        # 'lap',
                        # 'log',
                        # 'unif',
                        # 'Gaus_0',
                        # 'Gaus_1',
                        # 'Gaus_2',
                        # 'Gaus_3',
                        # 'Gaus_4',
                        # 'Gaus_5',
                        # 'Gaus_6',
                        # 'Gaus_7',
                        # 'Gaus_8',
                        # 'Gaus_9',
                        # 'Gaus_10'
        ]

        # have program run through each weight type defined
        for wt_index, type_ in enumerate(weight_types):
            print('The weight distribution type is: ' + type_ + '\n')
            log_file.write('The weight distribution type is: ' + type_ + '\n')
            if(args.t_conv != 3):
                weights = weight_switcher(type_, args.nf, args.filter_d, in_chan)
            else:
                weights = weight_switcher(type_, args.nf//2, args.filter_d, in_chan)
            # print(weights.size())
            start_time = time.time()

            # determines if CNN 1 conv or 2 conv layers
            if(args.t_conv == 1):
                model = CNN_1Lay(nf=args.nf, weights=weights, in_chan=in_chan).to(device)
            elif(args.t_conv == 2):
                model = CNN_2Lay(nf=args.nf, weights=weights, in_chan=in_chan).to(device)
            elif(args.t_conv == 3):
                model = CNN_1Lay_Split(nf=( args.nf//2 ), weights=weights, in_chan=in_chan).to(device)
            elif(args.t_conv == 4):
                model = CNN_1Lay_FullTrain(nf=args.nf, weights=weights, in_chan=in_chan).to(device)
            else:
                raise ValueError('Value should be either 1 or 2 or 3 or 4')

            # training and testing
            train_acc, train_time, loss_list, train_acc_list, test_acc_list = training(args.n_epochs, args.batch_size, train_data, model, args.lr, args.decay, log_file, args.delta, args.patience)
            test_acc, test_time = testing(test_data, model, log_file)
            end_time = time.time()

            time_taken = end_time - start_time

            csv_file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n'.format(dataset, args.t_conv, args.n_epochs, args.batch_size, args.lr, args.random_seed, args.decay, args.n_runs, runs, type_, train_acc, test_acc, train_time, test_time))


            plt.figure(wt_index)
            plt.subplots_adjust(hspace=0.5)
            plt.subplot(211)
            plt.title('Loss plot ' + type_ + ' distribution')
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.plot(loss_list)
            plt.subplot(212)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.plot(train_acc_list, 'r', label='training')
            plt.plot(test_acc_list, 'b', label='testing')
            plt.legend(loc='lower right')
            plt.savefig('./plots/' + today_time + '_' + args.log_name + '_' + type_ + '.png')
            plt.close('all')

            print('Program finished for ' + type_ + ' weight distribution at run: {:d}'.format(runs))
            print('Time taken for ' + type_ + ' weight distribution: {:.2f}mins \n'.format(time_taken/60))
            log_file.write('Program finished for ' + type_ + ' weight distribution at run: {:d} \n'.format(runs))
            log_file.write('Time taken for ' + type_ + ' weight distribution: {:.2f}s \n \n'.format(time_taken))
        csv_file.write('\n')

    log_file.close()
    csv_file.close()

    
