# Once again, I try the simple CNN on cifar10 ...

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

my_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=my_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=my_transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow_batch(xx, yy):
    n = len(yy)
    plt.figure(figsize=(1.2*n, 1.2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        npimg = xx[i].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        lbl = classes[yy[i]]
        plt.title(lbl)
        plt.axis('off')
    plt.show()

if False:
    # Visualize a batch
    dataiter = iter(train_loader)
    batch_x, batch_y = dataiter.next()
    imshow_batch(batch_x, batch_y)

#=================================================================
# Define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)
print('net =', net)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        batch_x, batch_y = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = net(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        # Print stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[{}, {}] loss {}'.format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.