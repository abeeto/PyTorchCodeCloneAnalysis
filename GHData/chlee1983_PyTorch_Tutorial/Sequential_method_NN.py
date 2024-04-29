import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]

    def __len__(self):
        return len(self.x)


ds = MyDataSet(x, y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

torch_summary = summary(model, torch.zeros(1, 2))
print(torch_summary)

loss_func = nn.MSELoss()
opt = SGD(model.parameters(), lr = 0.001)
loss_history = []
start = time.time()
for _ in range(120):
    for ix, iy in dl:
        opt.zero_grad()
        loss_value = loss_func(model(ix), iy)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end = time.time()
print(end - start)

'''validation dataset'''
val = [[8, 9], [10, 11], [1.5, 2.5]]
val1 = model(torch.tensor(val).float().to(device))
print(val1)

