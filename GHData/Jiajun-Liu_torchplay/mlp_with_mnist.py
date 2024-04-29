# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchvision
from simplemlp import TwoLayerNet
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

###########################################################################################################
# * Tutorial to create MLPs with specified number of layers, layer sizes, activation, and output
# * Using MNIST as a dataset to demonstrate DataLoader as well as model evaluation
# * Demonstrating how to off-load models and tensors/computation to GPU
###########################################################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# sizes is an array of input/hidden1/hidden2.../output, min length 2
def mlp_model_maker(sizes, activation=nn.LeakyReLU, output_trans=nn.Softmax(dim=1)):
    layers = []
    assert (len(sizes) > 1)
    for index in range(len(sizes) - 1):
        layer = nn.Linear(sizes[index], sizes[index + 1])
        layers.append(layer)

        if index < len(sizes) - 2 and activation is not None:
            layers.append(activation())

        if index == len(sizes) - 2 and output_trans is not None:
            layers.append(output_trans)

    model = nn.Sequential(*layers)
    return model


# * Mini-batch training, single batch pred with MNIST
# * Treating 2D images as 1D feature vectors with MLP
# * You can play with hyperparameters such as sizes, activation, output_trans, epoch, batch_size, loss, learning_rate
# to see different model behaviours
def mini_batch_mlp_with_mnist_on_gpu(sizes, train_loader, test_loader):
    # Construct our model by instantiating the class defined above
    # model = TwoLayerNetSM(D_in, H, D_out).to(device)
    model = mlp_model_maker(sizes, activation=nn.ReLU, output_trans=nn.LogSoftmax(dim=1)).to(device)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(2):
        avg_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.reshape((train_loader.batch_size, 28 * 28)).to(device)
            batch_y = batch_y.to(device)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)

            # Compute and print loss
            loss = F.nll_loss(y_pred, batch_y, reduction='sum')
            # loss = F.cross_entropy(y_pred, batch_y, reduction='sum')
            # print('loss', loss.item())
            avg_loss += loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("batch loss {}".format(loss))

        print("epoch {}, loss {}".format(epoch, avg_loss.item() / len(train_loader.dataset)))

    correct = 0.0
    test_loss = 0.0

    # forward only, no need to autograd
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.reshape((test_loader.batch_size, 28 * 28)).to(device)
            batch_y = batch_y.to(device)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)
            # Compute and print loss
            loss = F.nll_loss(y_pred, batch_y, reduction='sum').item()
            # loss = F.cross_entropy(y_pred, batch_y, reduction='sum').item()

            # extract predicted label and calculate accuracy
            y_pred = y_pred.data.max(1, keepdim=True)[1]
            correct += batch_y.eq(y_pred.data.view_as(batch_y)).sum()

            test_loss += loss

        print("test loss {}, accuracy {}".format(test_loss / len(test_loader.dataset),
                                                 correct / len(test_loader.dataset)))


if __name__ == '__main__':
    sizes, batch_size = [28 * 28, 512, 10], 100

    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../datasets/', train=True, download=True,
                                   transform=transform), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../datasets/', train=False, download=True,
                                   transform=transform), batch_size=batch_size, shuffle=True)

    mini_batch_mlp_with_mnist_on_gpu(sizes, train_loader, test_loader)
