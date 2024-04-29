import torch
import numpy as np

data = np.array([1,2,3,4])

t1 = torch.Tensor(data) #Uses global default dtype
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)

print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

print("Global Default dtype is: ", torch.get_default_dtype())

#Memory Copy vs Share
t3[0] = 0
t3[1] = 0
print(t3)
print(data)
print(t4)