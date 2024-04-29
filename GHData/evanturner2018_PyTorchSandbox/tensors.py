# Try out tensors as a start
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
# Notes to self:
# named VENV "pytorchOne"
# numpy, pytorch installed so far
# use "python" for the conda version of python

import torch
import numpy as np

data = [[1,2],[3,4]]
np_data = np.array(data)
tensor_data = torch.from_numpy(np_data)
print(tensor_data)

# attributes: shape (height, width), dtype (datatype), device
print(f"Tensor stored on: {tensor_data.device}")

# move to GPU
if torch.cuda.is_available():
    gpu_tensor = tensor_data.to("cuda")
    print(gpu_tensor.device)