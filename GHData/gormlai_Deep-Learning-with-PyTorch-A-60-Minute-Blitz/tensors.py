import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand}\n")

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(f"Tensor after slicing:\n{tensor}")

t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(f"Concatenated tensor:\n{t1}")

print(f"Multiplied tensor:\n{tensor.mul(tensor)}")
print(f"Multiplied tensor(alternative syntax):\n{tensor * tensor}")

print(f"Mat Multiplied tensor:\n{tensor.matmul(tensor.T)}\n")
print(f"Mat Multiplied tensor(alternative syntax):\n{tensor @ tensor.T}")

print(f"Tensor Before Add:\n {tensor}\n")
tensor.add_(5) # add 5 to all entries
print(f"Tensor:\n {tensor}\n")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out= n)
print(f"n: {n}")
print(f"t: {t}")
