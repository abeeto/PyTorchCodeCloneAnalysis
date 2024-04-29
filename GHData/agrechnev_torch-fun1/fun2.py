import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

#==============================================================
# CUDA
device = torch.device("cuda:0")
#==============================================================
# Load data + plots

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function: show 1 image
def imshow(img, lbl=None):
    img = img/2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.colorbar(), plt.axis('off')
    if lbl is not None:
        plt.title(lbl)
    plt.show()

# Visualize data
if False:
    # iterator
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(' '.join(classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

#==============================================================
#Define a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)
print('net =', net)

#optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train
if True:
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #stats
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[{}, {}] loss: {}'.format(epoch, i, running_loss/2000))
                running_loss = 0.
    print('Training finished !')
    # save data
    torch.save(net.state_dict(), 'cifar_net.pth')
else:
    #load data
    net.load_state_dict(torch.load('cifar_net.pth'))

# test 1 batch
if False:
    dataiter = iter(testloader)
    data = dataiter.next()
    inputs, labels = data[0].to(device), data[1].to(device)
    imshow(torchvision.utils.make_grid(inputs))
    print('GT =', ' '.join(classes[labels[j]] for j in range(4)))
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    print('PRED =', ' '.join(classes[predicted[j]] for j in range(4)))

# Test dataset
if True:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print('accuracy =', 100*correct/total)

# Breakdown by classes
if True:
    class_correct = [0. for i in range(10)]
    class_total = [0. for i in range(10)]
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('acc[{}] = {}'.format(classes[i], 100*class_correct[i]/class_total[i]))
