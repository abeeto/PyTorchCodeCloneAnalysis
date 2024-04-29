import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

x = np.linspace(start=1, stop=1000, num=100, dtype=np.float32)
x = x.reshape(-1, 1)
print(x.shape)
y = x**2 + 1
print(y.shape)
# create tensor dataset object - provide x and y
linData = TensorDataset(
    torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
)
for d in linData[0:3]:
    print(d)

# create batches and shuffle
data_loader = DataLoader(linData, batch_size=5, shuffle=True)

m = 0
for batch in data_loader:
    while m < 1:
        print(batch)
        m += 1
