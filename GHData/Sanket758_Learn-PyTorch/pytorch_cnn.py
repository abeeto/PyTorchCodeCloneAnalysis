"""
Description: Here we will implement a Convolutional Neural Network for Multiclass classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

import math
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 0: Prepare data
transformations = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
# Create dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transformations)
# Create Dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# images, labels = next(iter(train_loader))
# print(images.shape, labels.shape)
# # for i in range(6):
# #     plt.subplot(2, 3, i+1)
# #     plt.imshow(images[i][0], cmap='gray')
# # plt.show()

# cnn1 = nn.Conv2d(1, 64, 5)
# pool = nn.MaxPool2d(2, 2)
# cnn2 = nn.Conv2d(64, 128, 5)
# fc1 = nn.Linear(128*5*5, 64)
# fc2 = nn.Linear(64, 10)

# print(images.shape)
# x = cnn1(images)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# x = cnn2(x)
# print(x.shape)
# x = pool(x)
# print(x.shape)



# Step 1: Build Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cnn1 = nn.Conv2d(1, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.cnn2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*4*4, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.cnn1(x)))
        x = self.pool1(F.relu(self.cnn2(x)))
        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet()

# Define loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

# Step 2: Implement a Training Loop
num_epochs = 2
num_of_steps = len(train_loader)

for i in range(num_epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        out = model(images)

        # loss function
        loss = criterion(out, labels)

        # Backward pass
        loss.backward()

        # take optimizer step
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()

        if j%100==0:
            print(f"Epoch: {i}/{num_epochs} Step: {j}/{num_of_steps} Loss: {loss.item():.6f}")

print('Training completed')

n_samples = 0
n_correct = 0

n_class_samples = [0 for i in range(10)]
n_class_correct = [0 for i in range(10)]


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Overall Accuracy
        _, predictions = torch.max(output, 1)
        n_samples += labels.size(0)
        n_correct += (predictions==labels).sum().item()

        print(labels.shape, predictions)
        
        # Per Class accuracy
        for pred, label in zip(predictions, labels):
            if (label==pred):
                n_class_correct[label] += 1
            n_class_samples[label]+=1

acc = 100. * n_correct / n_samples
print('Overall Accuracy: ', acc)

print('Per class accuracy: ')
for i in range(10):
    acc = 100. * n_class_correct[i] / n_class_samples[i]
    print('Class: ', i, 'Accuracy: ', acc)

