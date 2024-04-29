# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini batches
import torchvision.datasets as datasets  # It has standard dataset we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

import numpy as np
from tqdm import tqdm  # For nice progress bar!

import logging

logging.basicConfig(level=logging.DEBUG)


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    logging.info(f'Saving checkpoint....')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    logging.info(f"Loading checkpoint.......")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Setup simple CNN models
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5000

# Load_dataset
train_dataset = datasets.MNIST(root='dataset/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# print('check train_data set.shape', train_dataset.shape())
test_dataset = datasets.MNIST(root='dataset/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network


def train_network(num_epochs=num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()


# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += np.array(predictions == y).sum()
            num_samples += predictions.size(0)

            model.train()
            return num_correct / num_samples


def main():
    print(
        f'Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}%'
    )
    print(
        f'accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}%')
    # #Initialize network
    # model = torchvision.models.vgg16(pretrained=False)
    # optimizer = optim.Adam(model.parameters())
    # checkpoint = {"state_dict": model.state_dict(),
    #               'optimizer': optimizer.state_dict()}
    #
    # # Try save checkpoint
    # save_checkpoint(checkpoint)
    #
    # # Try load checkpoint
    # load_checkpoint(torch.load('my_check_point.pth.tar'), model, optimizer)


if __name__ == '__main__':
    main()
