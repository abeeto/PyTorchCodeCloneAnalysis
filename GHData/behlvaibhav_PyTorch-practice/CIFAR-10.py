import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

_tasks = transforms.Compose((
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
))

# Load the dataset
cifar = CIFAR10('data', train=True, download=True, transform=_tasks)

# create training and validation split
split = int(0.8 * len(cifar))
index_list = list(range(len(cifar)))
train_idx, valid_idx = index_list[:split], index_list[split:]

# Create training and validation sampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

# Create iterator for training and validation datasets
train_loader = DataLoader(cifar, batch_size=256, sampler=tr_sampler)
valid_loader = DataLoader(cifar, batch_size=256, sampler=val_sampler)


# CNN

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 1024)  # reshaping
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


model = Model()

# loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      weight_decay=1e-6, momentum=0.9,
                      nesterov=True)

# Epochs
for epoch in range(1, 31):
    train_loss, valid_loss = [], []
    # train
    for data, target in train_loader:
        optimizer.zero_grad()
        # forward pass
        output = model(data)
        # loss
        loss = loss_function(output, target)
        # backward
        loss.backward()
        # optimization
        optimizer.step()
        train_loss.append(loss.item())
    # evaluation
    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())

# Predictions on validation set
data_iter = iter(valid_loader)
data, label = data_iter.next()
output = model(data)
_, pred_tensor = torch.max(output, 1)
pred = np.squeeze(pred_tensor.numpy())
print("Actual:", label[:10])
print("Predicted:", pred[:10])
