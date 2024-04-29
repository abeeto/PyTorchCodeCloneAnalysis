import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy array
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

# ones and randoms
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Ones Tensor: \n {x_rand} \n")

shape = (2, 3,)
zeros_t = torch.zeros(shape)
print(f"Zeros Tensor: \n {zeros_t} \n")

tensor = torch.ones(4, 4)

# if torch.cuda.is_available():
#     tensor = tensor.to('cuda')
#     print(f"Device tensor is stored on: {tensor.device}")

tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print(f"Mul: {tensor.mul(tensor)}")

print(f"Mat mul: {tensor.matmul(tensor)}")

print(f"Mat mul T: {tensor.matmul(tensor.T)}")  # @ also used

# op_  = in-place operations

tensor.add_(5)
print(tensor)