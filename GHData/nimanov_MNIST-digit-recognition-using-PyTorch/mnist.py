import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True), nn.BatchNorm2d(5),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(10), 
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True), nn.BatchNorm2d(20),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(40))
        
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))
        
    def forward(self, x):
    
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 40)
        x = self.classifier(x)
        return x

# Instantiation of the network
model = Net()

# Instantiation of the cross-entropy loss
loss = nn.CrossEntropyLoss()

# Instantiation of the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

nb_epochs = 10
for epoch in range(nb_epochs):
  losses = list()
  accuracies = list()
  model.train()
  for batch in train_loader:
    x, y = batch
    optimizer.zero_grad()

    # Computing the forward pass
    outputs = model(x)
        
    # Computing the loss function
    J = loss(outputs, y)
        
    # Computing the gradients
    J.backward() 
    
    # Updating the weights
    optimizer.step()
    losses.append(J.item())
    accuracies.append(y.eq(outputs.detach().argmax(dim=1)).float().mean())
  print(f'Epoch {epoch + 1}', end=', ')
  print(f'training loss: {torch.tensor(losses).mean() : .2f}', end=', ')
  print(f'training accuracy: {torch.tensor(accuracies).mean(): .2f}')

  losses = list()
  accuracies = list()
  model.eval()
  for batch in val_loader:
    x, y = batch

    # Computing the forward pass
    with torch.no_grad():
      outputs = model(x)
        
    # Computing the loss function
    J = loss(outputs, y)
    losses.append(J.item())
    accuracies.append(y.eq(outputs.detach().argmax(dim=1)).float().mean())
  print(f'Epoch {epoch + 1}', end=', ')
  print(f'validation loss: {torch.tensor(losses).mean() : .2f}', end=', ')
  print(f'validation accuracy: {torch.tensor(accuracies).mean(): .2f}')

