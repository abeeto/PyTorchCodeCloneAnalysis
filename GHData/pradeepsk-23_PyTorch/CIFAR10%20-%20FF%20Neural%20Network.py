import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10 dataset (images and labels)
train_dataset = CIFAR10(root='./Dataset/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = CIFAR10(root='./Dataset/CIFAR10', train=False, transform=transforms.ToTensor())

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)

# Fully connected neural network with one hidden layer
input_size = 3*32*32
output_size = 10

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, output_size))        
    
    def forward(self, x):
        return self.network(x)

# Model
model = NeuralNet(input_size, output_size).to(device)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 5
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Reshape images to (batch_size, input_size)  
        # Move tensors to the configured device
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i+1) % 500 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))