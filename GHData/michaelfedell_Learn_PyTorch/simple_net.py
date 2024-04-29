import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)  # 1 channel image => 6 channel output (3x3 conv)
        self.conv2 = nn.Conv2d(6, 16, 3)  # 6 channel input => 16 channel out by a 3x3 conv

        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 16 channels and images are 6x6 at this point => 120 channel out
        self.fc2 = nn.Linear(120, 84)  # 120 channel => 84
        self.fc3 = nn.Linear(84, 10)  # 84 channel => 10 outputs (possible digits)

    def forward(self, x):
        x = self.conv1(x)  # apply convolution
        x = F.relu(x)  # Relu activation
        x = F.max_pool2d(x, (2, 2))  # maxpool subsampling

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))  # flatten for FC layers

        x = F.relu(self.fc1(x))  # FC (dense) layer with relu activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # output layer (no activation)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

rand_in = torch.randn(1, 32, 32)  # 1 channel, 32 width, 32 height
batch = rand_in.unsqueeze(0)  # torch.nn accepts mini-batches as input so we need ot reshape our single image
print(rand_in.size(), '=>', batch.size())

out = net(batch)  # output layer for possible digit
print('Output:', out)

label = torch.randn(10)
label = label.view(1, -1)  # reshape target to same shape as input
print('"Label":', label)

criterion = nn.MSELoss()

loss = criterion(out, label)
print('Loss:', loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
out = net(batch)
loss = criterion(out, label)
loss.backward()
optimizer.step()
