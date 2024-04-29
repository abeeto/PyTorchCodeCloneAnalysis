import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

# net.zero_grad()
# out.backward(torch.randn(1, 10))
#
# output = net(input)
# target = Variable(torch.arange(1, 11))
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)

print(torch.arange(1, 11))

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
output = net(input)
criterion = nn.MSELoss()
target = Variable(torch.Tensor(1, 10))
# target = Variable(torch.arange(1, 11))
loss = criterion(output, target)
loss.backward()
optimizer.step()

