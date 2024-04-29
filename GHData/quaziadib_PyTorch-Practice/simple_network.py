# Imports 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Fully Connected Network class
class NN(nn.Module):
    def __init__(self, input_size, num_class): # inpsize=784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_class)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DeBug Code
# # model = NN(784, 10)
# # x = torch.randn(64, 784)
# # print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#print(device)
# Hyperoparameters
input_size = 784
num_class = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = NN(input_size, num_class).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)
        
        # Flatten All the examples
        data = data.reshape(data.shape[0], -1)
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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction==y).sum()
            num_samples += (prediction.size(0))
        print(f"Get {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)