from scripts.MNIST_modules import MNIST_modules
from scripts import MNIST_network
import torch


def main():

    # Defines network parameters
    batch_size_train = 128
    batch_size_test = 10000
    train_epochs = 5
    learning_rate = 1e-2
    momentum = 0

    visualize_training_data = False
    show_parameters = False
    train_network = True
    ce_torch = True

    # Imports MNIST dataset
    MNIST = MNIST_modules(batch_size_train=batch_size_train,
                          batch_size_test=batch_size_test,
                          train_epochs=train_epochs)
    MNIST.import_data()

    # Visualizes training data
    if visualize_training_data:
        MNIST.visualize_training_data()

    # Initializes network
    network = MNIST_network.MNIST_network()
    optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, momentum=momentum)

    # Prints weights and biases
    if show_parameters:
        MNIST.show_parameters(network)

    # Trains network
    if train_network:
        network = MNIST.train(network, optimizer, ce_torch)

    # Evaluates network
    MNIST.test(network)

    # Plots results of training
    MNIST.result()


if __name__ == "__main__":
    main()
