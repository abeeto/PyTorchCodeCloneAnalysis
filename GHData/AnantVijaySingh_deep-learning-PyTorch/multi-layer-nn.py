import torch


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


# Generate some data
torch.manual_seed(7)  # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]  # Number of input units, must match number of input features
n_hidden = 2  # Number of hidden units
n_output = 1  # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)

output_input_hidden = activation(torch.mm(features, W1) + B1)
output_hidden_output = activation(torch.mm(output_input_hidden, W2) + B2)

print(output_hidden_output)