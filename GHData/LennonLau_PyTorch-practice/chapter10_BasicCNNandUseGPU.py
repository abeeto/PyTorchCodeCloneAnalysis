import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


batch_size = 64
lr = 0.002
epochs = 100
predict_need = True
mnistfilepath = r'../data/MNIST'
mnisttestfilepath = r'../data/MNIST'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

trian_dataset = datasets.MNIST(root=mnistfilepath,
                               train=True,
                               download=True,
                               transform=transforms)

train_loader = DataLoader(dataset=trian_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root=mnisttestfilepath,
                              train=False,
                              download=True,
                              transform=transforms)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Cnet(torch.nn.Module):
    def __init__(self):
        super(Cnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 10, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Cnet()
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx+1) % 300 == 0:
            print('epoch={}, batch_idx={}, loss={}'.format(epoch+1,
                                                           batch_idx,
                                                           running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            max_, predict = torch.max(outputs.data, dim=1)
            correct += (predict == target).sum().item()
            total += target.size(0)
    print('acc on test:{}'.format(correct/total))


def save_lossimage(epoch_list, loss_list):
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        if (epoch+1) % 20 == 0:
            test()
