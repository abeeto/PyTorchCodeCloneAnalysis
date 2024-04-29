# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

from student_code import SimpleFCNet, train_model, test_model


# main function for training and testing
def main():
    batch_size = 32
    epochs = 5
    torch.manual_seed(0)

    ###################################
    # setup model, loss and optimizer #
    ###################################
    model = SimpleFCNet()
    training_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # set up transforms to transform the PIL Image to tensors
    # also keep the normalization same as Keras
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.0, 1./255.0)
    ])

    ################################
    # setup dataset and dataloader #
    ################################
    data_folder = './data'
    if not os.path.exists(data_folder):
        os.makedirs(os.path.expanduser(data_folder), exist_ok=True)

    train_set = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=data_folder, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    ################################
    # start the training           #
    ################################
    print("Training the model ...\n")
    for epoch in range(epochs):
        # train model for 1 epoch
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        # evaluate the model on test_set after this epoch
        test_model(model, test_loader, epoch)
    print("Finished Training")


if __name__ == '__main__':
    main()
