import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# MNIST dataset (images and labels)
train_dataset = MNIST(root='./Dataset/MNIST', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = MNIST(root='./Dataset/MNIST', train=False, transform=transforms.ToTensor())

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle = True)
test_dl = DataLoader(test_dataset, batch_size)

# Logistic regression model
input_size = 28*28
num_classes = 10
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# Train the model
epochs = 5
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)
        
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
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
