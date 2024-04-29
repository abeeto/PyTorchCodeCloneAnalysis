import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper

matplotlib.use('Qt4Agg', warn=False, force=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to the first hidden layer linear transformation
        self.hidden_1 = nn.Linear(784, 128)
        # Inputs to the second hidden layer linear transformation
        self.hidden_2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

        # Define sigmoid activation function, softmax, and cross-entropy loss
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.logsoftmax(x)

        return x

 # Get the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=False)])
trainset = datasets.FashionMNIST('FashionMNIST/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('FashionMNIST/', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
# Plot a sample image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

# Define and run network
model = Network()
# Define loss function
criterion = nn.NLLLoss()
# Optimizer require parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a vector with a length of 784
        images = images.view(images.shape[0], -1)
        # Clear the gradients, do this becuase gradients are accumulated
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(images)
        # Calculate loss
        loss = criterion(output, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Keep track of loss
        running_loss += loss.item()
    else:
        print('Training loss: {}'.format(running_loss/len(trainloader)))

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
img = img.resize_(1, 784)

# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model.forward(img)
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')