# By Oleksiy Grechnyev, IT-JIM on 6/25/20.
# Simple regression fun

import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'

# Network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

net = Net().to(device=device)
print(net)

# Data
x0 = np.random.rand(100)
y0 = np.sin(x0) * x0**3 + 3*x0 + np.random.rand(100)*0.8
x = torch.from_numpy(x0.reshape(-1, 1)).float().to(device=device)
y = torch.from_numpy(y0.reshape(-1, 1)).float().to(device=device)

# Optimizer and loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.4)
loss_func = torch.nn.MSELoss()

# Train
n_epoch = 100
for i in range(n_epoch):
    pred = net(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    w = net.layer.weight.item()
    b = net.layer.bias.item()
    print(f'{i} : loss={loss}, w={w}, b={b}')

pred0 = pred.cpu().detach().numpy()

# Show result
plt.scatter(x0, y0)
plt.plot(x0, pred0)
y_lin = x0*w + b
plt.plot(x0, y_lin, 'o')
plt.show()