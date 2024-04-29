"""
Training of neural network
"""
from datetime import datetime

import torch
from torch import nn, optim

from src import misc
from src import vis
from src.models import LeNet

BATCH_SIZE = 128
LABELS = [1, 2]


def perform_train_epoch(model, trainloader, criterion, optimizer, log_freq=10):
    """
    This function performs a training epoch on a trainloader. E.g. it performs gradient descent
    on mini-batches until all the samples of the dataset have been seen by the network.
    """

    model.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for idx, (inputs, labels) in enumerate(trainloader, 1):

        # We reset gradients to zero

        ##################
        # YOUR CODE HERE #
        ##################
        optimizer.zero_grad()

        # We perform the forward pass

        ##################
        # YOUR CODE HERE #
        ##################
        outputs = model(inputs)

        # We compute the loss

        ##################
        # YOUR CODE HERE #
        ##################
        loss = criterion(outputs, labels)

        # We perform the backward pass

        ##################
        # YOUR CODE HERE #
        ##################
        loss.backward()

        # We perform the optimization step

        ##################
        # YOUR CODE HERE #
        ##################
        optimizer.step()

        # We update total_loss, total_correct and total

        ##################
        # YOUR CODE HERE #
        ##################
        _, prediction = torch.max(outputs.data, 1)
        total += prediction.shape[0]
        total_correct += (prediction == labels).sum().item()
        total_loss += loss.item()

        if idx % log_freq == 0:
            train_loss = total_loss / idx
            train_accuracy = total_correct / total

    return total_loss/idx, total_correct/total

def evaluate_model(model, testloader, criterion):
    """
    This function evaluate the test error on a complete dataset.
    """

    with torch.no_grad():

        total_loss = 0
        total_correct = 0
        total = 0

        for idx, (inputs, labels) in enumerate(testloader):

            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            total_correct += (prediction == labels).sum().item()
            total += prediction.shape[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss/idx, total_correct/total


def train(epochs, lr=0.001, momentum=0.9, weight_decay=1e-4):
    """
    Trains a lenet on mnist for the indicated number of epochs
    """

    model = LeNet(LABELS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    trainloader, testloader = misc.load_mnist_data(kept_labels=LABELS, batch_size=BATCH_SIZE)
    train_loss_serie = []
    train_accuracy_serie = []
    test_loss_serie = []
    test_accuracy_serie = []

    for _ in range(epochs):

        train_loss, train_acc = perform_train_epoch(model, trainloader, criterion, optimizer)
        train_accuracy_serie.append(train_acc)
        train_loss_serie.append(train_loss)
        test_loss, test_acc = evaluate_model(model, testloader, criterion)
        test_accuracy_serie.append(test_acc)
        test_loss_serie.append(test_loss)
        print("Train accuracy: {} loss: {}".format(train_acc, train_loss))
        print("Test accuracy: {} loss: {}".format(test_acc, test_loss))

    torch.save(model.state_dict(), datetime.now().strftime(("checkpoints/final-%H:%M:%S.t7")))

    return model, train_loss_serie, train_accuracy_serie, test_loss_serie, test_accuracy_serie


if __name__ == "__main__":

    print("1) Training for a few epochs...")
    model, train_loss, train_acc, test_loss, test_acc = train(50)
    vis.plot_learning_curves(train_loss, test_loss, "Training Curves")

    print("Ok\n\n2) Previewing first layer kernels")
    vis.preview_kernels(list(model.parameters())[0], "Input kernels after learning")

