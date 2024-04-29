import torch

def activation(x):
    """
    Sigmoid activation function

    Arguments
    --------
    x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

torch.manual_seed(7) # generate fake date

features = torch.randn((1,5))# input data for the network

weights = torch.randn_like(features)

bias = torch.randn((1,1))

y = activation(torch.sum(features * weights) + bias)
print(y)

new_weights = weights.view(5, 1)

z = activation(torch.mm(features, new_weights) + bias)
print(z)
# ||
# ||
# ||
# ||
# ||
z = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(z)
