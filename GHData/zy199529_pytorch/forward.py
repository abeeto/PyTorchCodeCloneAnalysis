import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())
input = Variable(t.randn(1, 1, 32, 32))
out = net(input)
# print(out.size())
# net.zero_grad()
# out.backward(Variable(t.ones(1,10)))
target = Variable(t.arange(0, 10))
loss = nn.MSELoss((out, target))
print(loss)