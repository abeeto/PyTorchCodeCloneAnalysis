import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10 dataset (images and labels)
train_dataset = FashionMNIST(root='./Dataset/FashionMNIST', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = FashionMNIST(root='./Dataset/FashionMNIST', train=False, transform=transforms.ToTensor())

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1), # Default stride=1, padding=0.
        # Output pixel = {[Input Size(28 in this case) - K(Kernel) + 2P(Padding)] / S(Stride)} + 1. Default Stride=1, Padding=0.
        # out = [16, 28, 28]
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), # out = [32, 28, 28]
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # out = [32, 14, 14]

        nn.Conv2d(32, 64, kernel_size=3, padding=1), # out = [64, 14, 14]
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=2), # out = [128, 16, 16]
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # out = [128, 8, 8]

        nn.Conv2d(128, 256, kernel_size=3, padding=1), # out = [256, 8, 8]
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, padding=1), # out = [512, 8, 8]
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # out = [512, 4, 4],

        nn.Flatten(), 
        nn.Linear(512*4*4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10))
    
    def forward(self, x):
        return self.network(x)

# Model
model = ConvNet().to(device)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 5
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i+1) % 600 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))