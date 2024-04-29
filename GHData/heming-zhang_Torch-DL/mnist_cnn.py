import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mnist_cnn_dataset import MnistCNNTrain, MnistCNNTest

class MnistCNNLoad():
    def __init__(self, batchsize, worker):
        self.batchsize = batchsize
        self.worker = worker

    # Design a mini-batch gradient descent
    def load_data(self):
        batchsize = self.batchsize
        worker = self.worker
        train_dataset = MnistCNNTrain()
        train_loader = DataLoader(dataset = train_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        test_dataset = MnistCNNTest()
        test_loader = DataLoader(dataset = test_dataset,
                    batch_size = batchsize,
                    shuffle = True,
                    num_workers = worker)
        return train_loader, test_loader

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # parameters: 
        # input channels; 
        # output channels; 
        # filter size; 
        # stride; 
        # padding
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 5, 1, 4), # output space (16, 16, 16)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(16, 30, 5, 1, 4), # output shape (30, 10, 10)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(30, 40, 5, 1, 4), # output shape (40, 7, 7)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        self.out = torch.nn.Linear(40*7*7, 10)
    
    # Use ReLU as activation function
    def forward(self, x):
        x = self.conv1(x)
        self.after_conv1 = x
        x = self.conv2(x)
        self.after_conv2 = x
        x = self.conv3(x)
        self.after_conv3 = x
        x = x.view(x.size(0), -1) # flat (batch_size, 40 * 7 * 7)
        y_pred = self.out(x)
        return y_pred

class MnistCNNParameter():
    def __init__(self, 
                learning_rate, 
                momentum,
                cuda):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cuda = cuda
    
    def mnist_function(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        cuda = self.cuda
        if cuda:
            model = MnistCNN().cuda()
        else:
            model = MnistCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                    lr = learning_rate, 
                    momentum = momentum)
        return model, criterion, optimizer

class RunMnistCNN():
    def __init__(self, model, criterion, optimizer, 
                train_loader, test_loader, cuda):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cuda = cuda

    def train_mnist_cnn(self):
        model = self.model
        model.train()
        criterion = self.criterion
        optimizer = self.optimizer
        train_loader = self.train_loader
        cuda = self.cuda
        # Train data in certain epoch
        train_correct = 0
        for i, data in enumerate(train_loader):
            train_input, train_label = data
            train_input = np.array(train_input)
            train_label = np.array(train_label)
            # Wrap them in Variable
            train_input = Variable(torch.Tensor(train_input),
                        requires_grad = False)
            train_label = Variable(torch.LongTensor(train_label),
                        requires_grad = False)
            if cuda:
                train_input = train_input.cuda()
                train_label = train_label.cuda()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(train_input)
            # Compute the train_loss with Cross-Entropy loss
            train_loss = criterion(y_pred, train_label)
            # Clear gradients of all optimized class
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # Compute accuracy rate
            _, train_pred = torch.max(y_pred.data, 1)
            train_correct += (train_pred == train_label).sum()
        train_accuracy = float(train_correct) / len(train_loader.dataset)
        return float(train_loss), train_accuracy
    
    def test_mnist_cnn(self):
        model = self.model
        model.eval()
        criterion = self.criterion
        optimizer = self.optimizer
        test_loader = self.test_loader
        cuda = self.cuda
        # Test model accuracy after certain epoch
        test_correct = 0
        for i, data in enumerate(test_loader):
            test_input, test_label = data
            test_input = np.array(test_input)
            test_label = np.array(test_label)
            test_input = Variable(torch.Tensor(test_input), requires_grad = False)
            test_label = Variable(torch.LongTensor(test_label), requires_grad = False)
            if cuda:
                test_input = test_input.cuda()
                test_label = test_label.cuda()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(test_input)
            test_loss = criterion(y_pred, test_label)
            _, test_pred = torch.max(y_pred, 1)
            test_correct += (test_pred == test_label).sum()
        test_accuracy = float(test_correct) / len(test_loader.dataset) 
        return float(test_loss), test_accuracy

class MnistCNNPlot():
    def __init__(self, train_accuracy_rate, 
                test_accuracy_rate,
                epoch_num):
        self.train_accuracy_rate = train_accuracy_rate
        self.test_accuracy_rate = test_accuracy_rate
        self.epoch_num = epoch_num

    def plot(self):
        train_accuracy_rate = self.train_accuracy_rate
        test_accuracy_rate = self.test_accuracy_rate
        epoch_num = self.epoch_num
        # Plot train and test accuracy rate
        x = range(1, epoch_num + 1)
        plt.plot(x, train_accuracy_rate, 
                    label = "Mnist-CNN-Train-Accuracy-Rate")
        plt.plot(x, test_accuracy_rate, 
                    label = "Mnist-CNN-Test-Accuracy-Rate")
        plt.xlabel("Epoch Time")
        plt.ylabel("Accuracy Rate")
        plt.xticks(range(0, epoch_num + 1, 5))
        plt.title("Mnist-CNN-Accuracy-Graph")
        plt.legend()
        if os.path.isdir("./img") == False: 
            os.mkdir("./img")
        plt.savefig("./img/Mnist-CNN-Accuracy" + str(epoch_num) + ".png")
        plt.show()

class MnistConvImage():
    def __init__(self, one_loader, cuda):
        self.one_loader = one_loader
        self.cuda = cuda

    def show_conv_image(self):
        one_loader = self.one_loader
        cuda = self.cuda
        model = MnistCNN()
        if cuda:
            model = model.cuda()
        for data in one_loader:
            conv_image, conv_label = data
            conv_image = conv_image.type(torch.FloatTensor)
            conv_label = conv_label.type(torch.FloatTensor)
            conv_image = Variable(torch.FloatTensor(conv_image),
                        requires_grad = False)
            conv_label = Variable(torch.FloatTensor(conv_label),
                        requires_grad = False)
            if cuda:
                conv_image = conv_image.cuda()
                conv_label = conv_label.cuda()
            plt.figure(0)
            plt.imshow((conv_image.squeeze().cpu()), cmap = "gray")
            output = model(conv_image)
            for i in range(1, 7):
                plt.figure(i)
                plt.title("After conv1 {:d} featured map".format(i))
                # Show images after first convolution
                plt.imshow(model.after_conv1.squeeze().detach().cpu()[i - 1],
                             cmap = "gray")
                plt.colorbar()
            break
        plt.show()