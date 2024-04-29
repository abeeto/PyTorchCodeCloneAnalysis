from torch import nn, optim


def define_layers():
    """
    Build the layers of the neural network.
    :return: A sequential layer structure which has a 28x28 starting input layer and a 10 final output layer.
    """
    # Convolutional Size =  (Size - Kernel + 2 * Padding) / Stride + 1
    # Pooling Size =        (Size + 2 * Padding - Kernel) / Stride + 1
    # Flatten Size =        Last Convolutional Out Channels * Size^2
    return nn.Sequential(
        nn.Conv2d(1, 6, 3, padding=1),  # (28 - 3 + 2 * 1) / 1 + 1 = 28
        nn.ReLU(),
        nn.AvgPool2d(2, 2),     # (28 + 2 * 0 - 2) / 2 + 1 = 14
        nn.Conv2d(6, 20, 3),    # (14 - 3 + 2 * 0) / 1 + 1 = 12
        nn.ReLU(),
        nn.AvgPool2d(2, 2),     # (12 + 2 * 0 - 2) / 2 + 1 = 6
        nn.Flatten(),           # 20 * 6^2 = 720
        nn.Linear(720, 720),
        nn.ReLU(),
        nn.Linear(720, 720),
        nn.ReLU(),
        nn.Linear(720, 10)
    )


def define_loss():
    """
    Choose the loss calculation method for the network.
    :return: A loss method to use.
    """
    return nn.CrossEntropyLoss()


def define_optimizer(neural_network):
    """
    Choose the optimizer method for the network.
    :param neural_network: The neural network.
    :return: An optimizer method to use.
    """
    return optim.Adam(neural_network.parameters())
