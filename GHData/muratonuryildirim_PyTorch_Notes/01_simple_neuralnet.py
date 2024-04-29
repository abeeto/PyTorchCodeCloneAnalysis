import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_functions import check_accuracy

'''
0. prepare dataset
1. design model (input, output, forward pass)
2. initiate the model
3. define loss and optimizer
4. train the model (loss)
    - forward pass: compute prediction and loss
    - backward pass: update weights
5. test the model (accuracy)
'''

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0. prepare dataset
batch_size = 64
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

data_iter = iter(train_loader)
inputs, labels = next(data_iter)

# 1. design model (input, output, forward pass)
input_size = inputs.shape[-1] * inputs.shape[-2]
num_classes = len(torch.unique(labels))

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # get to correct shape (flatten an image (64,1,28,28) to (64,284))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# sanity check
model_X = NN(784, 10)  # number of features, num_classes
x = torch.randn(64, 784)  # num of observation, number of features
print(f'shape model_X: {model_X(x).shape}')

# 2. initiate the model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# 3. define loss and optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. train the model (loss)
num_epochs = 5
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

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} is {mean_loss:.5f}')


# 5. test the model (accuracy)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
