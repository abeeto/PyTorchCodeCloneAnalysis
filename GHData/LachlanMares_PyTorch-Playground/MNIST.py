import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 28*28
CATEGORIES = 10
BATCH_SIZE = 16
EPOCHS = 25


# Parameters
params = {'batch_size': BATCH_SIZE,
          'shuffle': True}


class LinearNet(nn.Module):
    def __init__(self, input_size=784,  categories=10):
        super().__init__()
        self.categories = categories
        self.input_size= input_size
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, self.categories)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


train_loader = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([torchvision.transforms.ToTensor()]))
test_loader = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([torchvision.transforms.ToTensor()]))

X_train = data.DataLoader(train_loader, **params)
X_test = data.DataLoader(test_loader, **params)

train_length = 0
y_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for example in X_train:
    Xe, Ye = example
    for y in Ye:
        y_dictionary[int(y)] += 1
        train_length += 1


print("Training Length", train_length)
print("Category Distribution:", y_dictionary)

linearnet = LinearNet(input_size=IMAGE_SIZE, categories=CATEGORIES)
optimiser = optim.Adam(linearnet.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    print("Epoch:", epoch)
    for batch in X_train:
        X, y = batch
        linearnet.zero_grad()
        output = linearnet(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimiser.step()
    print(loss)

correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch in X_test:
        X, y = batch
        output = linearnet(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct_predictions += 1
            total_samples += 1

print("Accuracy=",round((correct_predictions/total_samples)*100, 3))
