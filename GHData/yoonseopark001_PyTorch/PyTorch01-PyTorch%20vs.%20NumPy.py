# import PyTorch       # // import NumPy
import torch           # // import numpy as np

# random
torch.rand(n, m)       # // np.random.randn(3, 5)

# zeros and ones
torch.zeros(n, m)      # // np.zeros((n, m))
torch.ones(n, m)       # // np.ones((n, m))
torch.eye(n)           # // np.identity(2)

# shape
a = torch.rand(n, m)   # // a = np.random.randn(3, 5)
a.shape                # // a.shape

# matrix operations
torch.matmul(a, b)     # // np.dot(a, b)
a * b                  # // np.multiply(a, b)


======================================================

# Import torch
import torch

# Create random tensor of size 3 by 3
first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = first_tensor.shape

# Print the values of the tensor and its shape
print(first_tensor)
print(tensor_size)

======================================================

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix mulitplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)
