import torch
import numpy as np

# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)

# print(x_data)

# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

tensor1 = torch.rand(3,4)
tensor2 = torch.rand(3,4)

if torch.cuda.is_available():
    tensor1 = tensor1.to("cuda")
    tensor2 = tensor2.to("cuda")

print(tensor1)
print(tensor2)
print('______________________')

print(tensor1*tensor2)

agg = tensor1.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor1} \n")
tensor1.add_(5)
print(tensor1)