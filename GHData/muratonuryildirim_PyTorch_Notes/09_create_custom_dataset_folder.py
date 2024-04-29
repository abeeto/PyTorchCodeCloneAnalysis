import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1
num_epochs = 5

train_path = 'dataset/cats_dogs/cats_dogs_imbalance/cats_dogs_train'
test_path = 'dataset/cats_dogs/cats_dogs_imbalance/cats_dogs_test'

train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

#loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
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


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # evaluation state on, train state off

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    model.train()  # evaluation state off, train state on


# accuracy check
print('Checking accuracy on training dataset...')
check_accuracy(train_loader, model)
print('Checking accuracy on test dataset...')
check_accuracy(test_loader, model)