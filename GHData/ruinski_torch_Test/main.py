import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.tensor(np_array)

print(f"{x_data}")
print(f"{x_np}")