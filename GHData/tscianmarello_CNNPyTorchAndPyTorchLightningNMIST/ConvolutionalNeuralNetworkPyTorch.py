#Imports necessary:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
from torchviz import make_dot, make_dot_from_trace


#Convolutional Neural Network Class that inherits from the nn.Module
#Uses PyTorch's Conv2d() functions followed by PyTorch's ReLU() functions followed by MaxPool2d() functions.
#Selected hyperparameters of in_channels, out_channels, kernel_size, and padding were chosen based on the equation for
#Convolutional Neural Networks: O = ((W-K+2P)/S)+1, where O is the output height/length, W is the input height/length,
#K is the filter size, P is the padding, and S is the stride.

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding = 1)
        self.r1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding = 1)
        self.r2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.r3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=3)
        self.linear_trans = nn.Linear(64*1*1, 10)

    #Forward function that grabs the transformed results from the constructor and then passes them to the next function.
    def forward(self, x):
        result = self.conv1_1(x)
        result = self.r1(result)
        result = self.mp1(result)
        result = self.conv1_2(result)
        result = self.r2(result)
        result = self.mp2(result)
        result = self.conv1_3(result)
        result = self.r3(result)
        result = self.mp3(result)
        result = result.view(result.size(0), -1)
        result = self.linear_trans(result)
        return result

    #Returns the model architecture name
    def name(self):
        return "ConvolutionalNeuralNetwork"

#This is the training or testing function. It takes the iterations that should start at 0, the images, labels from the
#dataset, the number of epochs being run, and the training and testing loaders. To run the training dataset, first pass
#the testing_loader and then the training_loader. To run the testing dataset, first pass the training_loader and then
#the testing_loader.
#This function will output the current epoch % 100 (this number can be changed), the current loss of the associated
#epoch, and the accuracy.

def train_test(iter, images, labels, epochs, training_loader, testing_loader):
    #This is the training set.
    for epoch in range(1, epochs + 1):
        for i, (images, labels) in enumerate(testing_loader):
            correct_cnt = 0
            total_cnt = 0
            images = images.requires_grad_()
            optimizer.zero_grad()
            results1 = cnn(images)
            loss = loss_function(results1, labels)
            loss.backward()
            optimizer.step()
            iter = iter + 1
            if iter % 100 == 0:
                for images, labels in training_loader:
                    images = images.requires_grad_()
                    results2 = cnn(images)
                    _, pred_label = torch.max(results2.data, 1)
                    total_cnt = total_cnt + labels.size(0)
                    correct_cnt = correct_cnt + (pred_label==labels).sum()
                accuracy = 100*(correct_cnt/total_cnt)
                print('==>>> Current Epoch: {}. Current Training Loss: {:.6f}. Current Test Accuracy: {:.6f}'.format(
                    iter, loss.item(), accuracy))

if __name__ == '__main__':

    #You have to replace the directory in image_data with the directory where the downloaded image dataset is on
    #your computer.

    #Load the image data.
    image_data = "/home/thomas/notMNIST_small"

    #Detects GPU
    GPUS = 1
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and GPUS > 0) else 'cpu')

    #Transforms the code
    trsfrm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (1.0,))])

    #Separate the data into training and testing.
    training_set = datasets.MNIST(root = image_data, train = True, transform = trsfrm, download = True)
    testing_set = datasets.MNIST(root = image_data, train = False, transform = trsfrm, download = True)

    #Select appropriate batch size
    batch_size = 64

    #Load the training and testing data using DataLoader
    training_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
    testing_loader = torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size, shuffle = True)

    #Displays an example of the code to make sure that everything is running correctly.
    dataiter = iter(training_loader)
    images, labels = dataiter.next()
    images.size()

    #Displays an example of the code to make sure that everything is running correctly.
    image_grid = torchvision.utils.make_grid(images, normalize=True)
    plt.imshow(np.transpose(image_grid.numpy(), (1,2,0)), interpolation = 'nearest')
    plt.show()


    #Instantiate the ConvolutionalNeuralNetwork()
    cnn = ConvolutionalNeuralNetwork()
    cnn.to(device = DEVICE)
    #Select the loss function
    loss_function = nn.MultiMarginLoss()
    #Select the learning rate
    learning_rate = 0.01
    #Select the optimizer
    optimizer = torch.optim.SGD(cnn.parameters(), lr = learning_rate)

    #Visualizes the network architecture.
    make_dot(cnn(images.to(device=DEVICE)), params = dict(cnn.named_parameters()))

    #iterations
    iter = 0
    epochs = 2000

    #There is a training and testing set. If you want to run the testing set, comment out the training set. If you want
    #to run the training set, comment out the testing set. If one of the two is not commented out, it will take much
    #longer to increase the accuracy.

    train_test(iter, images, labels, epochs, training_loader, testing_loader)