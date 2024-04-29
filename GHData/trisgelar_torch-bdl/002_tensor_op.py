import numpy as np
import torch

numpy_arr = np.array([1, 2, 3])
numpy_arr

tensor = torch.from_numpy(numpy_arr)
tensor

numpy_arr[1] = 4
numpy_arr

tensor

initial_tensor = torch.rand(2,3)
initial_tensor

initial_tensor[0,2]
initial_tensor[0,1:]

resized_tensor = initial_tensor.view(6)
resized_tensor.shape

resized_tensor
initial_tensor[0,2] = 0.1111
resized_tensor

resized_tensor = initial_tensor.view(3, 2)
resized_tensor.shape

resized_tensor

resized_matrix = initial_tensor.view(-1,2)
resized_matrix.shape

resized_matrix = initial_tensor.view(-1,5)
resized_matrix.shape

sorted_tensor, sorted_index = torch.sort(initial_tensor, dim=0)

sorted_tensor
sorted_index



