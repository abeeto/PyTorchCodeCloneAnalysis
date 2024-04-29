import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper
import PyQt5
matplotlib.use('Qt5Agg', warn=False, force=True)


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
testset = datasets.FashionMNIST('FashionMNIST/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = Network()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    # Training pass
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad() # reset gradient
        images = images.view(images.shape[0], -1) # flatten the images
        log_ps = model(images) # forward pass
        loss = criterion(log_ps, labels) # compute loss
        loss.backward() # compute gradient
        optimizer.step() # update weights
        running_loss += loss.item() # record loss

    # Validation pass
    else:
        test_loss, accuracy = 0, 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                images = images.view(images.shape[0], -1)  # flatten the images
                log_ps = model(images) # generate scores
                test_loss += criterion(log_ps, labels) # compute loss
                ps = torch.exp(log_ps) # probabilities
                top_p, top_class = ps.topk(1, dim=1) # pick the class with the highest probability
                equals = top_class == labels.view(*top_class.shape) # compare to ground truth
                accuracy += torch.mean(equals.type(torch.FloatTensor)) # compute accuracy

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}.. ".format(accuracy/len(testloader)))

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend(frameon=False)