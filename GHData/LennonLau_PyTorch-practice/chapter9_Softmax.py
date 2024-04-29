import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


batch_size = 64
lr = 0.01
epochs =100
mnistfilepath = r'../data/MNIST/'
mnisttestfilepath = r'../data/MNIST/'


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root=mnistfilepath,
                              train=True,
                              download=True,
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root=mnisttestfilepath,
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        return self.linear5(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx+1) % 300 == 0:
            print('epoch={}, batch_idx={}, loss={}'.format(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad:
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            max_, predict = torch.max(outputs.data, dim=1)
            correct += (predict == target).sum().item()
            total += target.size(0)
    print('Accuracy on test dataset:{}'.format(correct/total))


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        if (epoch+1) % 20 == 0:
            test()
