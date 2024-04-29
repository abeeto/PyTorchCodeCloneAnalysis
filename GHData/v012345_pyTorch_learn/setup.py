import torch
import numpy

np_data = numpy.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)

print("\nnumpy", np_data, "\ntorch", torch_data,
      "\ntorch2numpy", torch_data.numpy())

# abs
data = [-1, -2, 3, 4]
tensor = torch.FloatTensor(data)

print("\nabs", "\nnumpy", numpy.abs(data), "\ntorch", torch.abs(tensor))

# sin
print("\nsin", numpy.sin(data), "\ntorch", torch.sin(tensor))

# mean
print("\nmean", numpy.mean(data), "\ntorch", torch.mean(tensor))

# Matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
print("\nMatrix multiplication", numpy.matmul(
    data, data), "\ntorch", torch.mm(tensor, tensor))
