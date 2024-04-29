# Imports 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision 

import sys
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)




# DeBug Code
# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)
# exit()

# Set device

#print(device)
# Hyperoparameters
in_channels = 1
num_class = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
load_model = True

# Load Data
train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
#model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network


for epoch in range(num_epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, target)

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()


# Check the performance 

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on trainning data")
    else:
        print("Checking accuracy on testing data")
    num_correct = 0
    num_samples = 0 
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction==y).sum()
            num_samples += (prediction.size(0))
        print(f"Get {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)