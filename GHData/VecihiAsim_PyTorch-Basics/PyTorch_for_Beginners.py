import torch

print("torch version : {}".format(torch.__version__))

# Create a Tensor with just ones in a column
a = torch.ones(5)
# Print the tensor we created
print(a)

# Create a Tensor with just zeros in a column
b = torch.zeros(5)
print(b)

c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)

d = torch.zeros(3, 2)
print(d)

e = torch.ones(3, 2)
print(e)

f = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f)

# 3D Tensor
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(g)

print(f.shape)

print(e.shape)

print(g.shape)

# Get element at index 2
print(c[2])

# All indices starting from 0

# Get element at row 1, column 0
print(f[1, 0])

# We can also use the following
print(f[1][0])

# Similarly for 3D Tensor
print(g[1, 0, 0])
print(g[1][0][0])

# All elements
print(f[:])

# All elements from index 1 to 2 (inclusive)
print(c[1:3])

# All elements till index 4 (exclusive)
print(c[:4])

# First row
print(f[0, :])

# Second column
print(f[:, 1])

int_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(int_tensor.dtype)

# What if we changed any one element to floating point number?
int_tensor = torch.tensor([[1, 2, 3], [4., 5, 6]])
print(int_tensor.dtype)
print(int_tensor)

# This can be overridden as follows
int_tensor = torch.tensor([[1, 2, 3], [4., 5, 6]], dtype=torch.int32)
print(int_tensor.dtype)
print(int_tensor)

# Import NumPy
import numpy as np

# Tensor to Array
f_numpy = f.numpy()
print(f_numpy)

# Array to Tensor
h = np.array([[8, 7, 6, 5], [4, 3, 2, 1]])
h_tensor = torch.from_numpy(h)
print(h_tensor)

# Create tensor
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[-1, 2, -3], [4, -5, 6]])

# Addition
print(tensor1 + tensor2)
# We can also use
print(torch.add(tensor1, tensor2))

# Subtraction
print(tensor1 - tensor2)
# We can also use
print(torch.sub(tensor1, tensor2))

# Multiplication
# Tensor with Scalar
print(tensor1 * 2)

# Tensor with another tensor
# Elementwise Multiplication
print(tensor1 * tensor2)

# Matrix multiplication
tensor3 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(torch.mm(tensor1, tensor3))

# Division
# Tensor with scalar
print(tensor1 / 2)

# Tensor with another tensor
# Elementwise division
print(tensor1 / tensor2)

# Create a tensor for CPU
# This will occupy CPU RAM
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

# Create a tensor for GPU
# This will occupy GPU RAM
tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

# This uses CPU RAM
tensor_cpu = tensor_cpu * 5

# This uses GPU RAM
# Focus on GPU RAM Consumption
tensor_gpu = tensor_gpu * 5

# Move GPU tensor to CPU
tensor_gpu_cpu = tensor_gpu.to(device='cpu')

# Move CPU tensor to GPU
tensor_cpu_gpu = tensor_cpu.to(device='cuda')
