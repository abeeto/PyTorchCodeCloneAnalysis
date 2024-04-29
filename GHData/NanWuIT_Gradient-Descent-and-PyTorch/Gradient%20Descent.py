#!/usr/bin/env python3
# Nan Wu

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Linear(nn.Module):
    """
    Linear (10) -> ReLU -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # make sure inputs are flattened

        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)  # preserve batch dim

        return x


class FeedForward(nn.Module):
    """
    Linear (256) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> ReLU-> LogSoftmax
    """
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(784, 256)          # the Linear value change to 256
        self.fc2 = nn.Linear(256, 64)           # the Linear value change to 64
        self.fc3 = nn.Linear(64, 10)           # the Linear value change to 10

    def forward(self, x): 
        x = x.view(x.shape[0], -1)              # make sure inputs are flattened

        x = F.relu(self.fc1(x))                 # change the value to 64

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))                 # change the value to 10
        x = F.log_softmax(x, dim = 1)           # preserve batch dim

        return x

class CNN(nn.Module):
    """
    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu -> Linear (10) -> LogSoftmax


    Reshaping outputs from the last conv layer prior to feeding them into the linear layers.
    """
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, stride = 1)    # Applies a 1D convolution over an input signal composed of several input planes
        self.maxp = nn.MaxPool2d(2,2)                   # MaxPool function
        self.conv2 = nn.Conv2d(10, 50, 5, stride = 1)   # Applies a 1D convolution over an input signal composed of several input planes
        self.fc1 = nn.Linear(800, 256)                  # the Linear value change to 256
        self.fc2 = nn.Linear(256, 10)                    # the Linear value change to 1

    def forward(self, x): 
        
        x = F.relu(self.conv1(x))                   # convolution it

        x = self.maxp(x)                            # pool it
        
        x = F.relu(self.conv2(x))                   # convolution it

        x = self.maxp(x)                            # pool it again

        x = x.view(x.shape[0], -1)                  # make sure inputs are flattened
        x = F.relu(self.fc1(x))                     # change the value to 256
        x = self.fc2(x)                     # change the value to 1
        x = F.log_softmax(x, dim = 1)               # preserve batch dim

        return x

class NNModel:
    def __init__(self, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        """

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

        # Download and load the test data
        testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        self.model = network

        self.lossfn = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

    def view_batch(self):
        """
        Return:
           1) A float32 numpy array (of dim [28*8, 28*8]), containing a tiling of the batch images,
           place the first 8 images on the first row, the second 8 on the second row, and so on

           2) An int 8x8 numpy array of labels corresponding to this tiling
        """
        
        for i, l in self.trainloader:                                         # loop over all the images and labels
            i, l = iter(self.trainloader).next()
            i = i.view(i.size(-4), i.size(-3), i.size(-2), i.size(-1))
            i = i.view(i.size(0), i.size(2), i.size(3))                       # free the redundant array
            i = i.view(int(np.sqrt(i.size(0))), int(np.sqrt(i.size(0))), i.size(1), i.size(2))  # processing the input as required format
            i = i.permute(0, 2, 1, 3)                                         # transforming the input as [8, 28, 8, 28]
            i = i.reshape(8 * 28, 8 * 28)                                     # reshape the image
        l = l.view(8, 8)                                                      # identify the labels
        return i.numpy(), l. numpy()

    def train_step(self):

        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self):
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                log_ps = self.model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples


def plot_result(results, names):
    """
    Take a 2D list/array, where row is accuracy at each epoch of training for given model, and
    names of each model, and display training curves
    """
    for i, r in enumerate(results):
        plt.plot(range(len(r)), r, label=names[i])
    plt.legend()
    plt.title("KMNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./part_2_plot.png")


def main():
    models = [Linear(), FeedForward(), CNN()]  # Change during development
    epochs = 10
    results = []

    # Can comment the below out during development
    images, labels = NNModel(Linear(), 0.003).view_batch()
    print(labels)
    plt.imshow(images, cmap="Greys")
    plt.show()

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        m = NNModel(model, 0.003)

        accuracies = [0]
        for e in range(epochs):
            m.train_epoch()
            accuracy = m.eval()
            print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
            accuracies.append(accuracy)
        results.append(accuracies)

    plot_result(results, [m.__class__.__name__ for m in models])


if __name__ == "__main__":
    main()
