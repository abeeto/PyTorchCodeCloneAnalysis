import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.mp1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.mp2 = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = x.view(-1, self.num_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat(self, x):
        sz = x.shape[1:]
        n = 1
        for s in sz:
            n *= s
        return n

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.01)

params = list(net.parameters())
inp = torch.randn(1, 1, 32, 32)
optimizer.zero_grad()
out = net(inp)
# Dummy target
target = torch.randn(10).view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(f'loss = {loss}')

# backprop
net.zero_grad()
print(f'before = {net.conv1.bias.grad}')
loss.backward()
print(f'after = {net.conv1.bias.grad}')
optimizer.step()
