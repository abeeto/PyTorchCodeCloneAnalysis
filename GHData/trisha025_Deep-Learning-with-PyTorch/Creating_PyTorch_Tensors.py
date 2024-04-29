import torch
import numpy as np

data = np.array([1,2,3])

t1 = torch.Tensor(data)             #class constructor
t2 = torch.tensor(data)             #factor function
t3 = torch.as_tensor(data)          #factor function
t4 = torch.from_numpy(data)         #factor function

print(t1)
print(t2)
print(t3)
print(t4)

print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

print(torch.get_default_dtype())    # to get default data type

print((torch.tensor(np.array([1,2,3]))).dtype)      #int as input and int as output

print(torch.tensor(np.array([1., 2., 3.])))     #float as input and float as output

print(torch.tensor(np.array([1,2,3]), dtype = torch.float64))       #explicity setting data type

#Memory Sharing vs Copying

data = np.array([1,2,3])
print(data)

t1 = torch.Tensor(data)
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)          #factor function
t4 = torch.from_numpy(data)

data[0] = 0
data[1] = 0
data[2] = 0

#creating additional copy of input data in memory
print(t1) 
print(t2)

#share data in memory
print(t3)
print(t4)

