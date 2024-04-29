import torch

def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


# Generate some data
# Set the random seed so things are predictable
torch.manual_seed(7)

# Features are 5 random normal variables

# creates a tensor with shape `(1, 5)`, one row and five columns, that contains values randomly distributed according
# to the normal distribution with a mean of zero and standard deviation of one.
features = torch.randn((1, 5))

# True weights for our data, random normal variables again
# creates another tensor with the same shape as `features`, again containing values from a normal distribution.
weights = torch.randn_like(features)

# and a true bias term
# creates a single value from a normal distribution.
bias = torch.randn((1, 1))

# TODO: Find the output

# PyTorch tensors can be added, multiplied, subtracted, etc, just like Numpy arrays. In general, you'll use PyTorch
# tensors pretty much the same way you'd use Numpy arrays. They come with some nice benefits though such as GPU
# acceleration which we'll get to later. For now, use the generated data to calculate the output of this simple
# single layer network.


# Simple linear summation method
# Now, make our labels from our data and true weights
outputSumMethod = activation(torch.sum(features * weights) + bias)
outputSumMethod = activation((features * weights).sum() + bias)

# Using matrix multiplication method
# We can use torch.mm() or torch.matmul(). Prefer mm as it does not do broadcasting and thus will alert us to issues
# with shape of the matrices

# Note: To see the shape of a tensor called tensor, use tensor.shape. If you're building neural networks,
# you'll be using this method often.
#
# There are a few options here: weights.reshape(), weights.resize_(), and weights.view().
#
# --- weights.reshape(a, b) will return a new tensor with the same data as weights with size (a, b) sometimes,
# and sometimes a clone, as in it copies the data to another part of memory.
#
# --- weights.resize_(a, b) returns the same
# tensor with a different shape. However, if the new shape results in fewer elements than the original tensor,
# some elements will be removed from the tensor (but not from memory). If the new shape results in more elements than
# the original tensor, new elements will be uninitialized in memory. Here I should note that the underscore at the
# end of the method denotes that this method is performed in-place. Here is a great forum thread to read more about
# in-place operations in PyTorch.
#
# --- weights.view(a, b) will return a new tensor with the same data as weights with size
# (a, b).

transposedWeights = weights.view((5, 1))
print(transposedWeights.shape)

outputMatrixMulti = torch.mm(features, weights.t()) + bias
outputMatrixMulti = torch.mm(features, transposedWeights) + bias

outputValue = activation(outputMatrixMulti)

print(outputValue)
