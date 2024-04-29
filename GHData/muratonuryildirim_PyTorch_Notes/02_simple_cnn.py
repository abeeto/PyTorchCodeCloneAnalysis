import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
from torch_functions import check_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
batch_size = 64
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

data_iter = iter(train_loader)
inputs, labels = next(data_iter)

# 1. design model (input, output, forward pass)
n_channels = inputs.shape[1]
input_size = inputs.shape[2] * inputs.shape[3]
num_classes = len(torch.unique(labels))


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=8,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2),
                                 stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))

        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# sanity check
model_X = CNN(1, 10)  # number of channels, num_classes
x = torch.randn(64, 1, 28, 28)
print(f'shape model_X: {model_X(x).shape}')

# initialize network
model = CNN(in_channels=n_channels, num_classes=num_classes).to(device)

# loss and optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
num_epochs = 2
print(f'Training started with {num_epochs} epoch(s)')

for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')


# accuracy check
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
