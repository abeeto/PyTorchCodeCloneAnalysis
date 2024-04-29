# tensor practice in PyTorch
import torch
import numpy as np

# tensors can be created directly from data
data = [[1,2],[3,4]]

# define x data as a tensor from the given data
x_data = torch.tensor(data)
print(f"Torch Created Tensor: \n {x_data}")
# print the properies of the created tensor
print(f"dtype: {x_data.dtype}, shape: {x_data.shape}")
# from raw data the tensor will inherit the properties of the
# data it is given

# test to see if shape can be converted to an arrray
print(f"shape as array: {np.array(x_data.shape)}")

# tensors can also be created from numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Torch from array tensor: {x_np} \n")

# create tensor with same properties as another tensor
x_ones = torch.ones_like(x_data)
# create tensor with same shape and different dtype
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(f"Tensor with same shape and dtype: {x_ones}")
print(f"dtype: {x_ones.dtype}, shape: {np.array(x_ones.shape)}\n")
print(f"Tensor with same shape and different dtype: {x_rand}")
print(f"dtype: {x_rand.dtype}, shape: {np.array(x_rand.shape)}\n")

# check location of tensor with .device
print(f"Random tensor location: {x_rand.device}")

# move location of tensor to different processing unit
if torch.cuda.is_available():
    x_rand = x_rand.to("cuda")
    print(f"Moved x_rand to: {x_rand.device}")

# demonstrate indexing and value assignment with tensors
ones = torch.ones(4,4)
print(f"Tensor's First Row: {ones[0]}")
print(f"Tensor's First Column: {ones[:,0]}")
print(f"Tensor's Last Column: {ones[...,-1]}")
# assign second row value of 0
ones[:,1]=0
print(f"Tensor with 2nd column equal to zero:\n{ones}")

# demonstrate concatonation of multiple tensors along a single dim
second_ones = torch.ones(4,4)
# assign third column to value of 5
second_ones[:,2] = 5
s1 = torch.cat([second_ones,second_ones,second_ones], dim = 1 )
print(f"Concatonated Tensor:\n{s1}")

# demonstrate matrix multiplication (not the same as a cross product)

# @ sign can be used for matrix multiplication
# ones.T is the reversed order tensor of ones
y1 = ones @ ones.T
print(f"y1 matrix product:\n{y1}")

# matmul also gives matrix product
y2 = ones.matmul(ones.T)
print(f"y2 matrix product:\n{y2}")

y3 = torch.rand_like(y1)
torch.matmul(ones,ones.T,out=y3)
print(f"y3 matrix product:\n{y3}")

# regular multiplication works the same as matrix multiplication with * and torch.mul()

# demonstrate aggrigation of tensor values
agg = ones.sum()
print(f"Summation of ones tensor:\n{agg}")

# demonstration conversion of single value tensor to python
# numerical value
agg_item = agg.item()
print(f"Standard value of tensor: {agg_item}, type: {type(agg_item)}")
